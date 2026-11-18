// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The chat template parsed by MiniJinja's parser (the Rust frontend's own Jinja dialect, set up as
//! its HF renderer sets it up: trim_blocks, lstrip_blocks, `{% generation %}`), then compiled into
//! an owned tree whose string literals are template text with their control tokens ([`Literals`]),
//! as the Python plugin's code generator makes them. Filters and tests are checked against the
//! ones the interpreter implements: a template using another fails to compile, loudly.

use std::rc::Rc;

use minijinja::machinery::{WhitespaceConfig, ast, parse};
use minijinja::syntax::SyntaxConfig;

use super::builtins;
use super::sym::{Kind, Sym, SymBuilder};
use super::value::{Dict, Value};

/// A compile error: the template uses something the encoder doesn't implement, or doesn't parse.
#[derive(Debug, Clone)]
pub(crate) struct CompileError(pub(crate) String);

type CResult<T> = Result<T, CompileError>;

/// The tokenizer's added-token spellings, found in the template's literal text (leftmost first,
/// longest at a position, as the Python plugin's regex alternation sorted by length).
pub(crate) struct Literals {
    matcher: Option<aho_corasick::AhoCorasick>,
    ids: Vec<u32>,
}

impl Literals {
    pub(crate) fn new(added: &[(String, u32)]) -> Self {
        let mut tokens: Vec<&(String, u32)> = added.iter().filter(|(s, _)| !s.is_empty()).collect();
        tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));
        let matcher = (!tokens.is_empty()).then(|| {
            aho_corasick::AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(tokens.iter().map(|(s, _)| s.as_str()))
                .expect("added-token automaton")
        });
        Self {
            matcher,
            ids: tokens.iter().map(|(_, id)| *id).collect(),
        }
    }

    /// The matches of added-token spellings in `text`: (start, end, id).
    pub(crate) fn find_all(&self, text: &str) -> Vec<(usize, usize, u32)> {
        let Some(matcher) = &self.matcher else {
            return Vec::new();
        };
        matcher
            .find_iter(text)
            .map(|m| (m.start(), m.end(), self.ids[m.pattern().as_usize()]))
            .collect()
    }

    /// A literal of the template: its added-token spellings are those tokens.
    pub(crate) fn sym(&self, text: &str) -> Sym {
        let mut b = SymBuilder::new();
        let mut pos = 0;
        for (a, e, id) in self.find_all(text) {
            b.push(Kind::Template, &text[pos..a]);
            b.push(Kind::Control(id), &text[a..e]);
            pos = e;
        }
        b.push(Kind::Template, &text[pos..]);
        b.finish()
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Stmt {
    Emit(Expr),
    Raw(Sym),
    For(Rc<ForLoop>),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    With(Vec<(Target, Expr)>, Vec<Stmt>),
    Set(Target, Expr),
    SetBlock(Target, Option<Expr>, Vec<Stmt>),
    FilterBlock(Expr, Vec<Stmt>),
    Macro(Rc<MacroDef>),
    CallBlock(Expr, Rc<MacroDef>),
    Do(Expr),
    Continue,
    Break,
}

#[derive(Debug)]
pub(crate) struct ForLoop {
    pub(crate) target: Target,
    pub(crate) iter: Expr,
    pub(crate) filter: Option<Expr>,
    pub(crate) recursive: bool,
    pub(crate) body: Vec<Stmt>,
    pub(crate) else_body: Vec<Stmt>,
}

#[derive(Debug)]
pub(crate) struct MacroDef {
    pub(crate) name: Rc<str>,
    pub(crate) args: Vec<Rc<str>>,
    /// Defaults of the last `defaults.len()` arguments.
    pub(crate) defaults: Vec<Expr>,
    pub(crate) body: Vec<Stmt>,
}

/// An assignment target.
#[derive(Debug, Clone)]
pub(crate) enum Target {
    Name(Rc<str>),
    Tuple(Vec<Target>),
    /// `ns.attr` (a namespace attribute).
    Attr(Expr, Rc<str>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinOp {
    And,
    Or,
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Rem,
    Pow,
    Concat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CmpOp {
    Eq,
    Ne,
    Lt,
    Lte,
    Gt,
    Gte,
    In,
    NotIn,
}

#[derive(Debug, Clone)]
pub(crate) enum Arg {
    Pos(Expr),
    Kw(Rc<str>, Expr),
    PosSplat(Expr),
    KwSplat(Expr),
}

#[derive(Debug, Clone)]
pub(crate) enum Expr {
    Var(Rc<str>),
    Const(Value),
    Slice(
        Box<Expr>,
        Option<Box<Expr>>,
        Option<Box<Expr>>,
        Option<Box<Expr>>,
    ),
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Compare(Box<Expr>, Vec<(CmpOp, Expr)>),
    Cond(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    Filter(Rc<str>, Option<Box<Expr>>, Vec<Arg>),
    Test(Rc<str>, Box<Expr>, Vec<Arg>),
    GetAttr(Box<Expr>, Rc<str>),
    GetItem(Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Arg>),
    List(Vec<Expr>),
    Map(Vec<(Expr, Expr)>),
}

/// Parse and compile a chat template.
pub(crate) fn compile(source: &str, literals: &Literals) -> CResult<Vec<Stmt>> {
    let whitespace = WhitespaceConfig {
        keep_trailing_newline: false,
        lstrip_blocks: true,
        trim_blocks: true,
    };
    let ast = parse(source, "chat", SyntaxConfig, whitespace)
        .map_err(|e| CompileError(format!("the chat template doesn't parse: {e}")))?;
    let mut c = Compiler { literals };
    match ast {
        ast::Stmt::Template(t) => c.stmts(&t.children),
        other => c.stmts(std::slice::from_ref(&other)),
    }
}

struct Compiler<'l> {
    literals: &'l Literals,
}

fn unsupported(what: &str) -> CompileError {
    CompileError(format!(
        "the nemotron4 structural encoder doesn't support {what} in a chat template"
    ))
}

impl Compiler<'_> {
    fn stmts(&mut self, stmts: &[ast::Stmt<'_>]) -> CResult<Vec<Stmt>> {
        stmts.iter().map(|s| self.stmt(s)).collect()
    }

    fn stmt(&mut self, stmt: &ast::Stmt<'_>) -> CResult<Stmt> {
        Ok(match stmt {
            ast::Stmt::Template(t) => Stmt::If(
                Expr::Const(Value::Bool(true)),
                self.stmts(&t.children)?,
                Vec::new(),
            ),
            ast::Stmt::EmitExpr(e) => Stmt::Emit(self.expr(&e.expr)?),
            ast::Stmt::EmitRaw(r) => Stmt::Raw(self.literals.sym(r.raw)),
            ast::Stmt::ForLoop(f) => Stmt::For(Rc::new(ForLoop {
                target: self.target(&f.target)?,
                iter: self.expr(&f.iter)?,
                filter: f.filter_expr.as_ref().map(|e| self.expr(e)).transpose()?,
                recursive: f.recursive,
                body: self.stmts(&f.body)?,
                else_body: self.stmts(&f.else_body)?,
            })),
            ast::Stmt::IfCond(i) => Stmt::If(
                self.expr(&i.expr)?,
                self.stmts(&i.true_body)?,
                self.stmts(&i.false_body)?,
            ),
            ast::Stmt::WithBlock(w) => Stmt::With(
                w.assignments
                    .iter()
                    .map(|(t, e)| Ok((self.target(t)?, self.expr(e)?)))
                    .collect::<CResult<_>>()?,
                self.stmts(&w.body)?,
            ),
            ast::Stmt::Set(s) => Stmt::Set(self.target(&s.target)?, self.expr(&s.expr)?),
            ast::Stmt::SetBlock(s) => Stmt::SetBlock(
                self.target(&s.target)?,
                s.filter.as_ref().map(|e| self.expr(e)).transpose()?,
                self.stmts(&s.body)?,
            ),
            ast::Stmt::AutoEscape(_) => {
                return Err(unsupported(
                    "an {% autoescape %} block (the encoder renders without escaping)",
                ));
            }
            ast::Stmt::FilterBlock(f) => {
                Stmt::FilterBlock(self.expr(&f.filter)?, self.stmts(&f.body)?)
            }
            ast::Stmt::Macro(m) => Stmt::Macro(Rc::new(self.macro_def(m)?)),
            ast::Stmt::CallBlock(c) => Stmt::CallBlock(
                Expr::Call(Box::new(self.expr(&c.call.expr)?), self.args(&c.call.args)?),
                Rc::new(self.macro_def(&c.macro_decl)?),
            ),
            ast::Stmt::Continue(_) => Stmt::Continue,
            ast::Stmt::Break(_) => Stmt::Break,
            ast::Stmt::Do(d) => Stmt::Do(Expr::Call(
                Box::new(self.expr(&d.call.expr)?),
                self.args(&d.call.args)?,
            )),
            _ => return Err(unsupported("template inheritance, includes or imports")),
        })
    }

    fn macro_def(&mut self, m: &ast::Macro<'_>) -> CResult<MacroDef> {
        let args = m
            .args
            .iter()
            .map(|a| match a {
                ast::Expr::Var(v) => Ok(Rc::from(v.id)),
                _ => Err(unsupported("a macro argument that is not a name")),
            })
            .collect::<CResult<_>>()?;
        Ok(MacroDef {
            name: Rc::from(m.name),
            args,
            defaults: m.defaults.iter().map(|e| self.expr(e)).collect::<CResult<_>>()?,
            body: self.stmts(&m.body)?,
        })
    }

    fn target(&mut self, e: &ast::Expr<'_>) -> CResult<Target> {
        Ok(match e {
            ast::Expr::Var(v) => Target::Name(Rc::from(v.id)),
            ast::Expr::List(l) => {
                Target::Tuple(l.items.iter().map(|x| self.target(x)).collect::<CResult<_>>()?)
            }
            ast::Expr::GetAttr(g) => Target::Attr(self.expr(&g.expr)?, Rc::from(g.name)),
            _ => return Err(unsupported("this assignment target")),
        })
    }

    fn args(&mut self, args: &[ast::CallArg<'_>]) -> CResult<Vec<Arg>> {
        args.iter()
            .map(|a| {
                Ok(match a {
                    ast::CallArg::Pos(e) => Arg::Pos(self.expr(e)?),
                    ast::CallArg::Kwarg(k, e) => Arg::Kw(Rc::from(*k), self.expr(e)?),
                    ast::CallArg::PosSplat(e) => Arg::PosSplat(self.expr(e)?),
                    ast::CallArg::KwargSplat(e) => Arg::KwSplat(self.expr(e)?),
                })
            })
            .collect()
    }

    fn boxed(&mut self, e: &ast::Expr<'_>) -> CResult<Box<Expr>> {
        Ok(Box::new(self.expr(e)?))
    }

    fn opt(&mut self, e: &Option<ast::Expr<'_>>) -> CResult<Option<Box<Expr>>> {
        e.as_ref().map(|e| self.boxed(e)).transpose()
    }

    fn constant(&self, v: &minijinja::Value) -> CResult<Value> {
        use minijinja::value::ValueKind;
        Ok(match v.kind() {
            ValueKind::None => Value::None,
            ValueKind::Undefined => Value::undefined("undefined"),
            ValueKind::Bool => Value::Bool(v.is_true()),
            ValueKind::Number if v.is_integer() => Value::Int(
                i64::try_from(v.clone()).map_err(|_| unsupported("an integer this large"))?,
            ),
            ValueKind::Number => Value::Float(
                f64::try_from(v.clone()).map_err(|_| unsupported("this number literal"))?,
            ),
            ValueKind::String => Value::Str(self.literals.sym(v.as_str().unwrap_or_default())),
            ValueKind::Seq => Value::list(
                v.try_iter()
                    .map_err(|_| unsupported("this constant"))?
                    .map(|x| self.constant(&x))
                    .collect::<CResult<_>>()?,
            ),
            ValueKind::Map => {
                let mut d = Dict::new();
                for k in v.try_iter().map_err(|_| unsupported("this constant"))? {
                    let val = v.get_item(&k).map_err(|_| unsupported("this constant"))?;
                    d.insert(self.constant(&k)?, self.constant(&val)?)
                        .map_err(|e| CompileError(e.to_string()))?;
                }
                Value::dict(d)
            }
            _ => return Err(unsupported("this constant")),
        })
    }

    fn expr(&mut self, e: &ast::Expr<'_>) -> CResult<Expr> {
        Ok(match e {
            ast::Expr::Var(v) => Expr::Var(Rc::from(v.id)),
            ast::Expr::Const(c) => Expr::Const(self.constant(&c.value)?),
            ast::Expr::Slice(s) => Expr::Slice(
                self.boxed(&s.expr)?,
                self.opt(&s.start)?,
                self.opt(&s.stop)?,
                self.opt(&s.step)?,
            ),
            ast::Expr::UnaryOp(u) => match u.op {
                ast::UnaryOpKind::Not => Expr::Not(self.boxed(&u.expr)?),
                ast::UnaryOpKind::Neg => Expr::Neg(self.boxed(&u.expr)?),
            },
            ast::Expr::BinOp(b) => {
                let (l, r) = (self.boxed(&b.left)?, self.boxed(&b.right)?);
                let cmp = |op| Expr::Compare(l.clone(), vec![(op, *r.clone())]);
                match b.op {
                    ast::BinOpKind::Eq => cmp(CmpOp::Eq),
                    ast::BinOpKind::Ne => cmp(CmpOp::Ne),
                    ast::BinOpKind::Lt => cmp(CmpOp::Lt),
                    ast::BinOpKind::Lte => cmp(CmpOp::Lte),
                    ast::BinOpKind::Gt => cmp(CmpOp::Gt),
                    ast::BinOpKind::Gte => cmp(CmpOp::Gte),
                    ast::BinOpKind::In => cmp(CmpOp::In),
                    ast::BinOpKind::ScAnd => Expr::Bin(BinOp::And, l, r),
                    ast::BinOpKind::ScOr => Expr::Bin(BinOp::Or, l, r),
                    ast::BinOpKind::Add => Expr::Bin(BinOp::Add, l, r),
                    ast::BinOpKind::Sub => Expr::Bin(BinOp::Sub, l, r),
                    ast::BinOpKind::Mul => Expr::Bin(BinOp::Mul, l, r),
                    ast::BinOpKind::Div => Expr::Bin(BinOp::Div, l, r),
                    ast::BinOpKind::FloorDiv => Expr::Bin(BinOp::FloorDiv, l, r),
                    ast::BinOpKind::Rem => Expr::Bin(BinOp::Rem, l, r),
                    ast::BinOpKind::Pow => Expr::Bin(BinOp::Pow, l, r),
                    ast::BinOpKind::Concat => Expr::Bin(BinOp::Concat, l, r),
                }
            }
            ast::Expr::Compare(c) => Expr::Compare(
                self.boxed(&c.expr)?,
                c.ops
                    .iter()
                    .map(|op| {
                        Ok((
                            match op.op {
                                ast::CompareOpKind::Eq => CmpOp::Eq,
                                ast::CompareOpKind::Ne => CmpOp::Ne,
                                ast::CompareOpKind::Lt => CmpOp::Lt,
                                ast::CompareOpKind::Lte => CmpOp::Lte,
                                ast::CompareOpKind::Gt => CmpOp::Gt,
                                ast::CompareOpKind::Gte => CmpOp::Gte,
                                ast::CompareOpKind::In => CmpOp::In,
                                ast::CompareOpKind::NotIn => CmpOp::NotIn,
                            },
                            self.expr(&op.expr)?,
                        ))
                    })
                    .collect::<CResult<_>>()?,
            ),
            ast::Expr::IfExpr(i) => Expr::Cond(
                self.boxed(&i.test_expr)?,
                self.boxed(&i.true_expr)?,
                self.opt(&i.false_expr)?,
            ),
            ast::Expr::Filter(f) => {
                if !builtins::has_filter(f.name) {
                    return Err(unsupported(&format!("the filter `{}`", f.name)));
                }
                Expr::Filter(Rc::from(f.name), self.opt(&f.expr)?, self.args(&f.args)?)
            }
            ast::Expr::Test(t) => {
                if !builtins::has_test(t.name) {
                    return Err(unsupported(&format!("the test `{}`", t.name)));
                }
                Expr::Test(Rc::from(t.name), self.boxed(&t.expr)?, self.args(&t.args)?)
            }
            ast::Expr::GetAttr(g) => Expr::GetAttr(self.boxed(&g.expr)?, Rc::from(g.name)),
            ast::Expr::GetItem(g) => {
                Expr::GetItem(self.boxed(&g.expr)?, self.boxed(&g.subscript_expr)?)
            }
            ast::Expr::Call(c) => Expr::Call(self.boxed(&c.expr)?, self.args(&c.args)?),
            ast::Expr::List(l) => {
                Expr::List(l.items.iter().map(|x| self.expr(x)).collect::<CResult<_>>()?)
            }
            ast::Expr::Map(m) => Expr::Map(
                m.keys
                    .iter()
                    .zip(m.values.iter())
                    .map(|(k, v)| Ok((self.expr(k)?, self.expr(v)?)))
                    .collect::<CResult<_>>()?,
            ),
        })
    }
}
