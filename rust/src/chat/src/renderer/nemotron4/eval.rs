// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The interpreter: runs the compiled template on [`Value`]s with jinja2's semantics
//! (transformers' sandboxed environment: scoping, attribute lookup, `Undefined`, loops, macros,
//! call blocks, namespaces), so the render is a [`Sym`] whose runs say where each character came
//! from.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::builtins;
use super::ir::{Arg, BinOp, CmpOp, Expr, ForLoop, MacroDef, Stmt, Target};
use super::sym::{Sym, SymBuilder, slice_bounds};
use super::value::{Callable, Dict, LoopState, VResult, Value, type_error};

/// A macro value: its definition and the scope it was defined in (and, for a call block's
/// `caller`, nothing else).
pub(crate) struct MacroValue {
    pub(crate) def: Rc<MacroDef>,
    pub(crate) closure: Rc<Scope>,
    /// The `caller` a call block hands the macro it calls.
    pub(crate) caller: Option<Value>,
}

/// One scope: the template's top level, a loop iteration, a `with` block, a macro call.
pub(crate) struct Scope {
    vars: RefCell<HashMap<Rc<str>, Value>>,
    parent: Option<Rc<Scope>>,
}

impl Scope {
    pub(crate) fn root() -> Rc<Scope> {
        Rc::new(Scope {
            vars: RefCell::new(HashMap::new()),
            parent: None,
        })
    }

    fn child(parent: &Rc<Scope>) -> Rc<Scope> {
        Rc::new(Scope {
            vars: RefCell::new(HashMap::new()),
            parent: Some(parent.clone()),
        })
    }

    fn get(&self, name: &str) -> Option<Value> {
        if let Some(v) = self.vars.borrow().get(name) {
            return Some(v.clone());
        }
        self.parent.as_ref().and_then(|p| p.get(name))
    }

    fn set(&self, name: Rc<str>, value: Value) {
        self.vars.borrow_mut().insert(name, value);
    }
}

/// How a statement list ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Normal,
    Break,
    Continue,
}

/// A `cycler(...)` or `joiner(...)` object's state.
pub(crate) enum Helper {
    Cycler { items: Vec<Value>, pos: usize },
    Joiner { sep: Value, used: bool },
}

const MAX_DEPTH: usize = 128;

/// The interpreter for one render.
pub(crate) struct Interp<'a> {
    pub(crate) globals: &'a HashMap<Rc<str>, Value>,
    pub(crate) helpers: RefCell<Vec<Helper>>,
    depth: RefCell<usize>,
}

/// Arguments of a call, evaluated.
#[derive(Default)]
pub(crate) struct Args {
    pub(crate) pos: Vec<Value>,
    pub(crate) kw: Vec<(Rc<str>, Value)>,
}

impl Args {
    pub(crate) fn kwarg(&self, name: &str) -> Option<&Value> {
        self.kw.iter().find(|(k, _)| &**k == name).map(|(_, v)| v)
    }

    /// The `i`-th positional argument or the keyword `name`.
    pub(crate) fn get(&self, i: usize, name: &str) -> Option<&Value> {
        self.pos.get(i).or_else(|| self.kwarg(name))
    }
}

impl<'a> Interp<'a> {
    pub(crate) fn new(globals: &'a HashMap<Rc<str>, Value>) -> Self {
        Self {
            globals,
            helpers: RefCell::new(Vec::new()),
            depth: RefCell::new(0),
        }
    }

    /// Render the template.
    pub(crate) fn render(&self, body: &[Stmt]) -> VResult<Sym> {
        let mut out = SymBuilder::new();
        let scope = Scope::root();
        self.exec(body, &scope, &mut out)?;
        Ok(out.finish())
    }

    fn lookup(&self, scope: &Rc<Scope>, name: &str) -> Value {
        scope.get(name).or_else(|| self.globals.get(name).cloned()).unwrap_or_else(|| {
            if builtins::is_global(name) {
                Value::Callable(Rc::new(Callable::Builtin(builtins::global_name(name))))
            } else {
                Value::undefined(name)
            }
        })
    }

    fn exec(&self, stmts: &[Stmt], scope: &Rc<Scope>, out: &mut SymBuilder) -> VResult<Flow> {
        for stmt in stmts {
            let flow = self.exec_one(stmt, scope, out)?;
            if flow != Flow::Normal {
                return Ok(flow);
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_one(&self, stmt: &Stmt, scope: &Rc<Scope>, out: &mut SymBuilder) -> VResult<Flow> {
        match stmt {
            Stmt::Raw(s) => out.push_sym(s),
            Stmt::Emit(e) => {
                let v = self.eval(e, scope)?;
                out.push_sym(&v.to_sym());
            }
            Stmt::If(cond, then, otherwise) => {
                let branch = if self.eval(cond, scope)?.truthy() {
                    then
                } else {
                    otherwise
                };
                return self.exec(branch, scope, out);
            }
            Stmt::For(f) => self.exec_for(f, scope, out, 0)?,
            Stmt::With(assignments, body) => {
                let inner = Scope::child(scope);
                for (target, expr) in assignments {
                    let v = self.eval(expr, scope)?;
                    self.assign(target, v, &inner)?;
                }
                return self.exec(body, &inner, out);
            }
            Stmt::Set(target, expr) => {
                let v = self.eval(expr, scope)?;
                self.assign(target, v, scope)?;
            }
            Stmt::SetBlock(target, filter, body) => {
                let mut captured = SymBuilder::new();
                self.exec(body, &Scope::child(scope), &mut captured)?;
                let mut v = Value::Str(captured.finish());
                if let Some(f) = filter {
                    v = self.apply_filter_chain(f, v, scope)?;
                }
                self.assign(target, v, scope)?;
            }
            Stmt::FilterBlock(filter, body) => {
                let mut captured = SymBuilder::new();
                self.exec(body, &Scope::child(scope), &mut captured)?;
                let v = self.apply_filter_chain(filter, Value::Str(captured.finish()), scope)?;
                out.push_sym(&v.to_sym());
            }
            Stmt::Macro(def) => {
                let m = MacroValue {
                    def: def.clone(),
                    closure: scope.clone(),
                    caller: None,
                };
                scope.set(
                    def.name.clone(),
                    Value::Callable(Rc::new(Callable::Macro(Rc::new(m)))),
                );
            }
            Stmt::CallBlock(call, def) => {
                let caller = Value::Callable(Rc::new(Callable::Macro(Rc::new(MacroValue {
                    def: def.clone(),
                    closure: scope.clone(),
                    caller: None,
                }))));
                let Expr::Call(callee, args) = call else {
                    unreachable!("compiled as a call")
                };
                let f = self.eval(callee, scope)?;
                let args = self.eval_args(args, scope)?;
                let v = self.call_with_caller(&f, args, Some(caller))?;
                out.push_sym(&v.to_sym());
            }
            Stmt::Do(call) => {
                self.eval(call, scope)?;
            }
            Stmt::Continue => return Ok(Flow::Continue),
            Stmt::Break => return Ok(Flow::Break),
        }
        Ok(Flow::Normal)
    }

    fn exec_for(
        &self,
        f: &Rc<ForLoop>,
        scope: &Rc<Scope>,
        out: &mut SymBuilder,
        depth0: usize,
    ) -> VResult<()> {
        if f.recursive {
            return Err(type_error(
                "the nemotron4 structural encoder doesn't support recursive loops",
            ));
        }
        let iterable = self.eval(&f.iter, scope)?;
        let mut items = iterable.iterate()?;
        if let Some(filter) = &f.filter {
            let mut kept = Vec::with_capacity(items.len());
            for item in items {
                let s = Scope::child(scope);
                self.assign(&f.target, item.clone(), &s)?;
                if self.eval(filter, &s)?.truthy() {
                    kept.push(item);
                }
            }
            items = kept;
        }
        if items.is_empty() {
            self.exec(&f.else_body, &Scope::child(scope), out)?;
            return Ok(());
        }
        let n = items.len();
        for (i, item) in items.iter().enumerate() {
            let s = Scope::child(scope);
            self.assign(&f.target, item.clone(), &s)?;
            s.set(
                Rc::from("loop"),
                Value::Loop(Rc::new(LoopState {
                    index0: i,
                    length: n,
                    depth0,
                    previtem: (i > 0).then(|| items[i - 1].clone()),
                    nextitem: items.get(i + 1).cloned(),
                    changed_last: RefCell::new(None),
                })),
            );
            match self.exec(&f.body, &s, out)? {
                Flow::Break => break,
                Flow::Continue | Flow::Normal => {}
            }
        }
        Ok(())
    }

    fn assign(&self, target: &Target, value: Value, scope: &Rc<Scope>) -> VResult<()> {
        match target {
            Target::Name(name) => scope.set(name.clone(), value),
            Target::Tuple(targets) => {
                let items = value.iterate()?;
                if items.len() != targets.len() {
                    return Err(type_error(format!(
                        "too {} values to unpack (expected {})",
                        if items.len() > targets.len() {
                            "many"
                        } else {
                            "few"
                        },
                        targets.len()
                    )));
                }
                for (t, v) in targets.iter().zip(items) {
                    self.assign(t, v, scope)?;
                }
            }
            Target::Attr(obj, name) => match self.eval(obj, scope)? {
                Value::Namespace(ns) => {
                    ns.borrow_mut().insert(Value::template_str(name), value)?;
                }
                _ => {
                    return Err(type_error(
                        "cannot assign attribute on non-namespace object",
                    ));
                }
            },
        }
        Ok(())
    }

    fn apply_filter_chain(&self, filter: &Expr, value: Value, scope: &Rc<Scope>) -> VResult<Value> {
        // `{% filter a | b(x) %}` / `{% set x | a %}`: the chain with the captured text innermost
        match filter {
            Expr::Filter(name, inner, args) => {
                let v = match inner {
                    Some(inner) => self.apply_filter_chain(inner, value, scope)?,
                    None => value,
                };
                let args = self.eval_args(args, scope)?;
                builtins::filter(self, name, v, args)
            }
            _ => Err(type_error("expected a filter")),
        }
    }

    pub(crate) fn eval_args(&self, args: &[Arg], scope: &Rc<Scope>) -> VResult<Args> {
        let mut out = Args::default();
        for a in args {
            match a {
                Arg::Pos(e) => out.pos.push(self.eval(e, scope)?),
                Arg::Kw(k, e) => out.kw.push((k.clone(), self.eval(e, scope)?)),
                Arg::PosSplat(e) => out.pos.extend(self.eval(e, scope)?.iterate()?),
                Arg::KwSplat(e) => match self.eval(e, scope)? {
                    Value::Dict(d) => {
                        for (k, v) in d.iter() {
                            out.kw.push((Rc::from(k.to_sym().as_str()), v.clone()));
                        }
                    }
                    other => {
                        return Err(type_error(format!(
                            "argument after ** must be a mapping, not {}",
                            other.type_name()
                        )));
                    }
                },
            }
        }
        Ok(out)
    }

    pub(crate) fn eval(&self, e: &Expr, scope: &Rc<Scope>) -> VResult<Value> {
        Ok(match e {
            Expr::Var(name) => self.lookup(scope, name),
            Expr::Const(v) => v.clone(),
            Expr::List(items) => {
                Value::list(items.iter().map(|x| self.eval(x, scope)).collect::<VResult<_>>()?)
            }
            Expr::Map(pairs) => {
                let mut d = Dict::new();
                for (k, v) in pairs {
                    d.insert(self.eval(k, scope)?, self.eval(v, scope)?)?;
                }
                Value::dict(d)
            }
            Expr::Not(x) => Value::Bool(!self.eval(x, scope)?.truthy()),
            Expr::Neg(x) => match self.eval(x, scope)? {
                Value::Int(i) => Value::Int(-i),
                Value::Bool(b) => Value::Int(-i64::from(b)),
                Value::Float(f) => Value::Float(-f),
                other => {
                    return Err(type_error(format!(
                        "bad operand type for unary -: '{}'",
                        other.type_name()
                    )));
                }
            },
            Expr::Bin(op, l, r) => match op {
                BinOp::And => {
                    let a = self.eval(l, scope)?;
                    if a.truthy() { self.eval(r, scope)? } else { a }
                }
                BinOp::Or => {
                    let a = self.eval(l, scope)?;
                    if a.truthy() { a } else { self.eval(r, scope)? }
                }
                BinOp::Concat => {
                    let (a, b) = (self.eval(l, scope)?, self.eval(r, scope)?);
                    Value::Str(a.to_sym().concat(&b.to_sym()))
                }
                _ => {
                    let (a, b) = (self.eval(l, scope)?, self.eval(r, scope)?);
                    binop(*op, &a, &b)?
                }
            },
            Expr::Compare(first, ops) => {
                let mut left = self.eval(first, scope)?;
                for (op, rexpr) in ops {
                    let right = self.eval(rexpr, scope)?;
                    if !compare(*op, &left, &right)? {
                        return Ok(Value::Bool(false));
                    }
                    left = right;
                }
                Value::Bool(true)
            }
            Expr::Cond(test, then, otherwise) => {
                if self.eval(test, scope)?.truthy() {
                    self.eval(then, scope)?
                } else {
                    match otherwise {
                        Some(x) => self.eval(x, scope)?,
                        None => Value::undefined("the else branch of an if expression"),
                    }
                }
            }
            Expr::Filter(name, inner, args) => {
                let v = match inner {
                    Some(x) => self.eval(x, scope)?,
                    None => Value::undefined("filter input"),
                };
                let args = self.eval_args(args, scope)?;
                builtins::filter(self, name, v, args)?
            }
            Expr::Test(name, x, args) => {
                let v = self.eval(x, scope)?;
                let args = self.eval_args(args, scope)?;
                Value::Bool(builtins::test(self, name, &v, &args)?)
            }
            Expr::GetAttr(obj, name) => {
                let o = self.eval(obj, scope)?;
                getattr(&o, name)?
            }
            Expr::GetItem(obj, key) => {
                let o = self.eval(obj, scope)?;
                let k = self.eval(key, scope)?;
                getitem(&o, &k)?
            }
            Expr::Slice(obj, start, stop, step) => {
                let o = self.eval(obj, scope)?;
                let bound = |b: &Option<Box<Expr>>| -> VResult<Option<i64>> {
                    match b {
                        None => Ok(None),
                        Some(x) => match self.eval(x, scope)? {
                            Value::None => Ok(None),
                            v => v.as_int().map(Some).ok_or_else(|| {
                                type_error("slice indices must be integers or None")
                            }),
                        },
                    }
                };
                let (a, b, c) = (bound(start)?, bound(stop)?, bound(step)?);
                slice_value(&o, a, b, c)?
            }
            Expr::Call(callee, args) => {
                let f = self.eval(callee, scope)?;
                let args = self.eval_args(args, scope)?;
                self.call_with_caller(&f, args, None)?
            }
        })
    }

    pub(crate) fn call(&self, f: &Value, args: Args) -> VResult<Value> {
        self.call_with_caller(f, args, None)
    }

    fn call_with_caller(&self, f: &Value, args: Args, caller: Option<Value>) -> VResult<Value> {
        match f {
            Value::Callable(c) => match c.as_ref() {
                Callable::Macro(m) => self.call_macro(m, args, caller),
                Callable::Builtin(name) => builtins::call_global(self, name, args, caller),
                Callable::Method(receiver, name) => {
                    builtins::call_method(self, receiver, name, args)
                }
                Callable::Helper(id) => builtins::call_helper(self, *id, args),
            },
            Value::Undefined(name) => Err(type_error(format!("'{name}' is undefined"))),
            other => Err(type_error(format!(
                "'{}' object is not callable",
                other.type_name()
            ))),
        }
    }

    fn call_macro(&self, m: &MacroValue, args: Args, caller: Option<Value>) -> VResult<Value> {
        {
            let mut d = self.depth.borrow_mut();
            if *d >= MAX_DEPTH {
                return Err(type_error("maximum recursion depth exceeded in a macro"));
            }
            *d += 1;
        }
        let result = (|| {
            let def = &m.def;
            let s = Scope::child(&m.closure);
            let n_args = def.args.len();
            let first_default = n_args - def.defaults.len();
            let mut varargs = Vec::new();
            for (i, v) in args.pos.into_iter().enumerate() {
                if i < n_args {
                    s.set(def.args[i].clone(), v);
                } else {
                    varargs.push(v);
                }
            }
            let mut kwargs = Dict::new();
            for (k, v) in args.kw {
                if def.args.contains(&k) {
                    s.set(k, v);
                } else {
                    kwargs.insert(Value::template_str(&k), v)?;
                }
            }
            for (i, name) in def.args.iter().enumerate() {
                if s.vars.borrow().contains_key(name) {
                    continue;
                }
                let v = if i >= first_default {
                    self.eval(&def.defaults[i - first_default], &s)?
                } else {
                    Value::undefined(name)
                };
                s.set(name.clone(), v);
            }
            s.set(Rc::from("varargs"), Value::list(varargs));
            s.set(Rc::from("kwargs"), Value::dict(kwargs));
            if let Some(c) = caller.or_else(|| m.caller.clone()) {
                s.set(Rc::from("caller"), c);
            }
            let mut out = SymBuilder::new();
            self.exec(&def.body, &s, &mut out)?;
            Ok(Value::Str(out.finish()))
        })();
        *self.depth.borrow_mut() -= 1;
        result
    }
}

/// jinja2's `environment.getattr`: an attribute (a method, a namespace's or loop's attribute),
/// else the item; missing, `Undefined`. Attributes of `Undefined` are an error.
pub(crate) fn getattr(obj: &Value, name: &str) -> VResult<Value> {
    if let Some(m) = builtins::method_of(obj, name) {
        return Ok(m);
    }
    Ok(match obj {
        Value::Dict(d) => d.get_str(name).cloned().unwrap_or_else(|| Value::undefined(name)),
        Value::Namespace(ns) => {
            ns.borrow().get_str(name).cloned().unwrap_or_else(|| Value::undefined(name))
        }
        Value::Loop(l) => loop_attr(l, name),
        Value::Undefined(base) => return Err(type_error(format!("'{base}' is undefined"))),
        _ => Value::undefined(name),
    })
}

/// jinja2's `environment.getitem`: the item, else the attribute (string keys); missing,
/// `Undefined`.
pub(crate) fn getitem(obj: &Value, key: &Value) -> VResult<Value> {
    let found = match (obj, key) {
        (Value::Dict(d), k) => d.get(k).cloned(),
        (Value::Namespace(ns), k) => ns.borrow().get(k).cloned(),
        (Value::List(items) | Value::Tuple(items), k) if k.as_int().is_some() => {
            let n = items.len() as i64;
            let i = k.as_int().expect("checked");
            let i = if i < 0 { i + n } else { i };
            (0..n).contains(&i).then(|| items[i as usize].clone())
        }
        (Value::Str(s), k) if k.as_int().is_some() => {
            s.index_char(k.as_int().expect("checked")).ok().map(Value::Str)
        }
        (Value::Undefined(base), _) => return Err(type_error(format!("'{base}' is undefined"))),
        _ => None,
    };
    if let Some(v) = found {
        return Ok(v);
    }
    if let Value::Str(k) = key {
        let attr = getattr(obj, k.as_str())?;
        return Ok(attr);
    }
    Ok(Value::undefined("item"))
}

fn loop_attr(l: &Rc<LoopState>, name: &str) -> Value {
    let i = l.index0 as i64;
    let n = l.length as i64;
    match name {
        "index" => Value::Int(i + 1),
        "index0" => Value::Int(i),
        "revindex" => Value::Int(n - i),
        "revindex0" => Value::Int(n - i - 1),
        "first" => Value::Bool(i == 0),
        "last" => Value::Bool(i == n - 1),
        "length" => Value::Int(n),
        "depth" => Value::Int(l.depth0 as i64 + 1),
        "depth0" => Value::Int(l.depth0 as i64),
        "previtem" => l.previtem.clone().unwrap_or_else(|| Value::undefined("previtem")),
        "nextitem" => l.nextitem.clone().unwrap_or_else(|| Value::undefined("nextitem")),
        "cycle" | "changed" => Value::Callable(Rc::new(Callable::Method(
            Value::Loop(l.clone()),
            Rc::from(name),
        ))),
        _ => Value::undefined(name),
    }
}

fn slice_value(o: &Value, a: Option<i64>, b: Option<i64>, c: Option<i64>) -> VResult<Value> {
    Ok(match o {
        Value::Str(s) => Value::Str(s.slice(a, b, c)?),
        Value::List(items) | Value::Tuple(items) => {
            let step = c.unwrap_or(1);
            if step == 0 {
                return Err(type_error("slice step cannot be zero"));
            }
            let (start, stop) = slice_bounds(items.len() as i64, a, b, step);
            let mut out = Vec::new();
            let mut i = start;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                out.push(items[i as usize].clone());
                i += step;
            }
            if matches!(o, Value::Tuple(_)) {
                Value::tuple(out)
            } else {
                Value::list(out)
            }
        }
        Value::Undefined(base) => return Err(type_error(format!("'{base}' is undefined"))),
        other => {
            return Err(type_error(format!(
                "'{}' object is not subscriptable",
                other.type_name()
            )));
        }
    })
}

/// Python's arithmetic operators.
pub(crate) fn binop(op: BinOp, a: &Value, b: &Value) -> VResult<Value> {
    if let (Value::Undefined(name), _) | (_, Value::Undefined(name)) = (a, b) {
        return Err(type_error(format!("'{name}' is undefined")));
    }
    let unsupported = || {
        type_error(format!(
            "unsupported operand type(s) for {}: '{}' and '{}'",
            match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::FloorDiv => "//",
                BinOp::Rem => "%",
                BinOp::Pow => "** or pow()",
                _ => "?",
            },
            a.type_name(),
            b.type_name()
        ))
    };
    match (op, a, b) {
        (BinOp::Add, Value::Str(x), Value::Str(y)) => return Ok(Value::Str(x.concat(y))),
        (BinOp::Add, Value::List(x), Value::List(y)) => {
            return Ok(Value::list(x.iter().chain(y.iter()).cloned().collect()));
        }
        (BinOp::Add, Value::Tuple(x), Value::Tuple(y)) => {
            return Ok(Value::tuple(x.iter().chain(y.iter()).cloned().collect()));
        }
        (BinOp::Mul, Value::Str(s), n) | (BinOp::Mul, n, Value::Str(s)) if n.as_int().is_some() => {
            return Ok(Value::Str(s.repeat(n.as_int().expect("checked"))));
        }
        (BinOp::Mul, Value::List(items), n) | (BinOp::Mul, n, Value::List(items))
            if n.as_int().is_some() =>
        {
            let k = n.as_int().expect("checked").max(0) as usize;
            return Ok(Value::list(
                items.iter().cloned().cycle().take(items.len() * k).collect(),
            ));
        }
        (BinOp::Rem, Value::Str(fmt), args) => return builtins::percent_format(fmt, args),
        (BinOp::Add, Value::Str(_), other)
        | (BinOp::Add, Value::List(_), other)
        | (BinOp::Add, Value::Tuple(_), other)
            if a.type_name() != other.type_name() =>
        {
            return Err(type_error(format!(
                "can only concatenate {} (not \"{}\") to {}",
                a.type_name(),
                if other.as_sym().is_some() {
                    "str"
                } else {
                    other.type_name()
                },
                a.type_name()
            )));
        }
        _ => {}
    }
    let (Some(x), Some(y)) = (a.as_number(), b.as_number()) else {
        return Err(unsupported());
    };
    let ints = a.as_int().zip(b.as_int());
    Ok(match op {
        BinOp::Add => match ints {
            Some((i, j)) => {
                Value::Int(i.checked_add(j).ok_or_else(|| type_error("integer overflow"))?)
            }
            None => Value::Float(x + y),
        },
        BinOp::Sub => match ints {
            Some((i, j)) => {
                Value::Int(i.checked_sub(j).ok_or_else(|| type_error("integer overflow"))?)
            }
            None => Value::Float(x - y),
        },
        BinOp::Mul => match ints {
            Some((i, j)) => {
                Value::Int(i.checked_mul(j).ok_or_else(|| type_error("integer overflow"))?)
            }
            None => Value::Float(x * y),
        },
        BinOp::Div => {
            if y == 0.0 {
                return Err(type_error("division by zero"));
            }
            Value::Float(x / y)
        }
        BinOp::FloorDiv => match ints {
            Some((_, 0)) => return Err(type_error("integer division or modulo by zero")),
            Some((i, j)) => {
                Value::Int(i.div_euclid(j) - i64::from((i.rem_euclid(j) != 0) && (j < 0)))
            }
            None if y == 0.0 => return Err(type_error("float floor division by zero")),
            None => Value::Float((x / y).floor()),
        },
        BinOp::Rem => match ints {
            Some((_, 0)) => return Err(type_error("integer division or modulo by zero")),
            Some((i, j)) => Value::Int(((i % j) + j) % j),
            None if y == 0.0 => return Err(type_error("float modulo")),
            None => Value::Float(x - y * (x / y).floor()),
        },
        BinOp::Pow => match ints {
            Some((i, j)) if j >= 0 => Value::Int(
                i.checked_pow(u32::try_from(j).map_err(|_| type_error("integer overflow"))?)
                    .ok_or_else(|| type_error("integer overflow"))?,
            ),
            _ => Value::Float(x.powf(y)),
        },
        _ => return Err(unsupported()),
    })
}

/// Python's comparison operators (`in` and `not in` included).
pub(crate) fn compare(op: CmpOp, a: &Value, b: &Value) -> VResult<bool> {
    Ok(match op {
        CmpOp::Eq => a.py_eq(b)?,
        CmpOp::Ne => !a.py_eq(b)?,
        CmpOp::Lt => a.py_lt(b)?,
        CmpOp::Gt => b.py_lt(a)?,
        CmpOp::Lte => a.py_lt(b)? || a.py_eq(b)?,
        CmpOp::Gte => b.py_lt(a)? || a.py_eq(b)?,
        CmpOp::In => contains(b, a)?,
        CmpOp::NotIn => !contains(b, a)?,
    })
}

/// `needle in container`.
pub(crate) fn contains(container: &Value, needle: &Value) -> VResult<bool> {
    Ok(match container {
        Value::Str(s) => match needle {
            Value::Str(n) => s.contains(n)?,
            other => {
                return Err(type_error(format!(
                    "'in <string>' requires string as left operand, not {}",
                    other.type_name()
                )));
            }
        },
        Value::List(items) | Value::Tuple(items) => {
            for item in items.iter() {
                if item.py_eq(needle)? {
                    return Ok(true);
                }
            }
            false
        }
        Value::Dict(d) => d.get(needle).is_some(),
        Value::Namespace(ns) => ns.borrow().get(needle).is_some(),
        Value::Undefined(_) => false,
        other => {
            return Err(type_error(format!(
                "argument of type '{}' is not iterable",
                other.type_name()
            )));
        }
    })
}
