// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! jinja2's filters, tests and globals, transformers' additions (`tojson`, `raise_exception`,
//! `strftime_now`), and Python's `str` / `dict` / `list` methods, on [`Value`]s: what chat
//! templates use. Strings keep their runs through every one of them.

use std::rc::Rc;

use super::eval::{Args, Helper, Interp, compare, contains, getattr};
use super::format;
use super::ir::CmpOp;
use super::json;
use super::sym::{Kind, Sym, SymBuilder};
use super::value::{Callable, Dict, VResult, Value, type_error};

const FILTERS: &[&str] = &[
    "abs",
    "attr",
    "batch",
    "capitalize",
    "center",
    "count",
    "d",
    "default",
    "dictsort",
    "e",
    "escape",
    "first",
    "float",
    "forceescape",
    "format",
    "indent",
    "int",
    "items",
    "join",
    "last",
    "length",
    "list",
    "lower",
    "map",
    "max",
    "min",
    "reject",
    "rejectattr",
    "replace",
    "reverse",
    "round",
    "safe",
    "select",
    "selectattr",
    "slice",
    "sort",
    "string",
    "sum",
    "title",
    "tojson",
    "trim",
    "truncate",
    "unique",
    "upper",
    "wordcount",
];

const TESTS: &[&str] = &[
    "boolean",
    "callable",
    "defined",
    "divisibleby",
    "eq",
    "equalto",
    "escaped",
    "even",
    "false",
    "float",
    "ge",
    "greaterthan",
    "gt",
    "in",
    "integer",
    "iterable",
    "le",
    "lessthan",
    "lower",
    "lt",
    "mapping",
    "ne",
    "none",
    "number",
    "odd",
    "sameas",
    "sequence",
    "string",
    "true",
    "undefined",
    "upper",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
];

const GLOBALS: &[&str] = &[
    "range",
    "dict",
    "namespace",
    "cycler",
    "joiner",
    "raise_exception",
    "strftime_now",
    "__hf_generation",
];

const STR_METHODS: &[&str] = &[
    "capitalize",
    "casefold",
    "center",
    "count",
    "endswith",
    "find",
    "format",
    "format_map",
    "index",
    "isalnum",
    "isalpha",
    "isascii",
    "isdecimal",
    "isdigit",
    "isidentifier",
    "islower",
    "isnumeric",
    "isprintable",
    "isspace",
    "istitle",
    "isupper",
    "join",
    "ljust",
    "lower",
    "lstrip",
    "partition",
    "removeprefix",
    "removesuffix",
    "replace",
    "rfind",
    "rindex",
    "rjust",
    "rpartition",
    "rsplit",
    "rstrip",
    "split",
    "splitlines",
    "startswith",
    "strip",
    "swapcase",
    "title",
    "upper",
    "zfill",
];

const DICT_METHODS: &[&str] = &[
    "items",
    "keys",
    "values",
    "get",
    "copy",
    "update",
    "pop",
    "popitem",
    "setdefault",
    "clear",
];

const LIST_METHODS: &[&str] = &[
    "append", "extend", "insert", "remove", "pop", "clear", "index", "count", "sort", "reverse",
    "copy",
];

pub(crate) fn has_filter(name: &str) -> bool {
    FILTERS.contains(&name)
}

pub(crate) fn has_test(name: &str) -> bool {
    TESTS.contains(&name)
}

pub(crate) fn is_global(name: &str) -> bool {
    GLOBALS.contains(&name)
}

pub(crate) fn global_name(name: &str) -> &'static str {
    GLOBALS.iter().find(|g| **g == name).copied().expect("checked by is_global")
}

/// A bound method `obj.name`, if `name` is a method of the value's Python type.
pub(crate) fn method_of(obj: &Value, name: &str) -> Option<Value> {
    let known = match obj {
        Value::Str(_) => STR_METHODS.contains(&name),
        Value::Dict(_) => DICT_METHODS.contains(&name),
        Value::List(_) => LIST_METHODS.contains(&name),
        Value::Tuple(_) => matches!(name, "index" | "count"),
        _ => false,
    };
    known.then(|| Value::Callable(Rc::new(Callable::Method(obj.clone(), Rc::from(name)))))
}

fn arg_sym<'v>(v: &'v Value, what: &str) -> VResult<&'v Sym> {
    v.as_sym()
        .ok_or_else(|| type_error(format!("{what} must be str, not {}", v.type_name())))
}

fn opt_int(v: Option<&Value>) -> VResult<Option<i64>> {
    match v {
        None | Some(Value::None) => Ok(None),
        Some(v) => v.as_int().map(Some).ok_or_else(|| {
            type_error(format!(
                "'{}' object cannot be interpreted as an integer",
                v.type_name()
            ))
        }),
    }
}

fn unsafe_call(obj: &Value, name: &str) -> VResult<Value> {
    Err(type_error(format!(
        "access to attribute '{name}' of '{}' object is unsafe.",
        obj.type_name()
    )))
}

/// Call a bound method.
pub(crate) fn call_method(
    interp: &Interp<'_>,
    obj: &Value,
    name: &str,
    args: Args,
) -> VResult<Value> {
    match obj {
        Value::Str(s) => str_method(s, name, &args),
        Value::Dict(d) => match name {
            "items" => Ok(Value::list(
                d.iter().map(|(k, v)| Value::tuple(vec![k.clone(), v.clone()])).collect(),
            )),
            "keys" => Ok(Value::list(d.keys().cloned().collect())),
            "values" => Ok(Value::list(d.iter().map(|(_, v)| v.clone()).collect())),
            "get" => {
                let key = args
                    .pos
                    .first()
                    .ok_or_else(|| type_error("get expected at least 1 argument"))?;
                Ok(d.get(key)
                    .cloned()
                    .unwrap_or_else(|| args.pos.get(1).cloned().unwrap_or(Value::None)))
            }
            "copy" => Ok(Value::dict(d.as_ref().clone())),
            _ => unsafe_call(obj, name),
        },
        Value::List(items) | Value::Tuple(items) => match name {
            "count" => {
                let x = args
                    .pos
                    .first()
                    .ok_or_else(|| type_error("count() takes exactly one argument"))?;
                let mut n = 0;
                for item in items.iter() {
                    if item.py_eq(x)? {
                        n += 1;
                    }
                }
                Ok(Value::Int(n))
            }
            "index" => {
                let x = args
                    .pos
                    .first()
                    .ok_or_else(|| type_error("index expected at least 1 argument"))?;
                for (i, item) in items.iter().enumerate() {
                    if item.py_eq(x)? {
                        return Ok(Value::Int(i as i64));
                    }
                }
                Err(type_error("x not in list"))
            }
            "copy" => Ok(Value::list(items.as_ref().clone())),
            _ => unsafe_call(obj, name),
        },
        Value::Loop(l) => match name {
            "cycle" => {
                if args.pos.is_empty() {
                    return Err(type_error("no items for cycling given"));
                }
                Ok(args.pos[l.index0 % args.pos.len()].clone())
            }
            "changed" => {
                let mut last = l.changed_last.borrow_mut();
                let now = args.pos.clone();
                let changed = match last.as_ref() {
                    None => true,
                    Some(prev) => {
                        prev.len() != now.len()
                            || prev
                                .iter()
                                .zip(now.iter())
                                .any(|(a, b)| !a.py_eq(b).unwrap_or(false))
                    }
                };
                *last = Some(now);
                Ok(Value::Bool(changed))
            }
            _ => Err(type_error(format!("loop has no method {name}"))),
        },
        _ => {
            let _ = interp;
            Err(type_error(format!(
                "'{}' object has no attribute '{name}'",
                obj.type_name()
            )))
        }
    }
}

fn str_method(s: &Sym, name: &str, args: &Args) -> VResult<Value> {
    let a0 = args.pos.first();
    let sym_arg = |i: usize, what: &str| -> VResult<Option<&Sym>> {
        match args.pos.get(i) {
            None | Some(Value::None) => Ok(None),
            Some(v) => arg_sym(v, what).map(Some),
        }
    };
    let bool_v = |b: bool| Ok(Value::Bool(b));
    let text = s.as_str();
    match name {
        "upper" => Ok(Value::Str(s.upper())),
        "lower" | "casefold" => Ok(Value::Str(s.lower())),
        "title" => Ok(Value::Str(s.title())),
        "capitalize" => Ok(Value::Str(s.capitalize())),
        "swapcase" => Ok(Value::Str(s.swapcase())),
        "strip" | "lstrip" | "rstrip" => {
            let chars = sym_arg(0, "strip arg")?;
            Ok(Value::Str(s.strip(
                chars,
                name != "rstrip",
                name != "lstrip",
            )?))
        }
        "split" | "rsplit" => {
            let sep = match args.get(0, "sep") {
                None | Some(Value::None) => None,
                Some(v) => Some(arg_sym(v, "sep")?),
            };
            let maxsplit = opt_int(args.get(1, "maxsplit"))?.unwrap_or(-1);
            let parts = if name == "split" {
                s.split(sep, maxsplit)?
            } else {
                s.rsplit(sep, maxsplit)?
            };
            Ok(Value::list(parts.into_iter().map(Value::Str).collect()))
        }
        "splitlines" => {
            let keep = args.get(0, "keepends").is_some_and(Value::truthy);
            Ok(Value::list(
                s.splitlines(keep).into_iter().map(Value::Str).collect(),
            ))
        }
        "startswith" | "endswith" => {
            let affixes = match a0 {
                Some(Value::Str(x)) => vec![x.clone()],
                Some(Value::Tuple(xs) | Value::List(xs)) => {
                    xs.iter().map(|x| arg_sym(x, name).cloned()).collect::<VResult<_>>()?
                }
                Some(other) => {
                    return Err(type_error(format!(
                        "{name} first arg must be str or a tuple of str, not {}",
                        other.type_name()
                    )));
                }
                None => return Err(type_error(format!("{name}() takes at least 1 argument"))),
            };
            let (start, end) = (opt_int(args.pos.get(1))?, opt_int(args.pos.get(2))?);
            for a in &affixes {
                if s.affix(a, start, end, name == "endswith")? {
                    return bool_v(true);
                }
            }
            bool_v(false)
        }
        "find" | "rfind" | "index" | "rindex" | "count" => {
            let needle = arg_sym(
                a0.ok_or_else(|| type_error(format!("{name}() takes at least 1 argument")))?,
                name,
            )?;
            let (start, end) = (opt_int(args.pos.get(1))?, opt_int(args.pos.get(2))?);
            let i = match name {
                "find" | "index" => s.find(needle, start, end)?,
                "count" => return Ok(Value::Int(s.count(needle, start, end)?)),
                _ => s.rfind(needle, start, end)?,
            };
            if i < 0 && name.ends_with("index") {
                return Err(type_error("substring not found"));
            }
            Ok(Value::Int(i))
        }
        "replace" => {
            let old = arg_sym(
                args.get(0, "old").ok_or_else(|| type_error("replace expected 2 arguments"))?,
                "replace",
            )?;
            let new = arg_sym(
                args.get(1, "new").ok_or_else(|| type_error("replace expected 2 arguments"))?,
                "replace",
            )?;
            let count = opt_int(args.get(2, "count"))?.unwrap_or(-1);
            Ok(Value::Str(s.replace(old, new, count)?))
        }
        "join" => {
            let items =
                a0.ok_or_else(|| type_error("join() takes exactly one argument"))?.iterate()?;
            let parts = items
                .iter()
                .enumerate()
                .map(|(k, x)| {
                    x.as_sym().cloned().ok_or_else(|| {
                        type_error(format!(
                            "sequence item {k}: expected str instance, {} found",
                            x.type_name()
                        ))
                    })
                })
                .collect::<VResult<Vec<_>>>()?;
            Ok(Value::Str(s.join(&parts)))
        }
        "partition" | "rpartition" => {
            let sep = arg_sym(
                a0.ok_or_else(|| type_error("partition() takes exactly one argument"))?,
                name,
            )?;
            let (a, b, c) = s.partition(sep, name == "rpartition")?;
            Ok(Value::tuple(vec![
                Value::Str(a),
                Value::Str(b),
                Value::Str(c),
            ]))
        }
        "removeprefix" | "removesuffix" => {
            let affix = arg_sym(
                a0.ok_or_else(|| type_error(format!("{name}() takes exactly one argument")))?,
                name,
            )?;
            let at_end = name == "removesuffix";
            if !affix.is_empty() && s.affix(affix, None, None, at_end)? {
                let n = s.len_chars();
                let k = affix.len_chars();
                return Ok(Value::Str(if at_end {
                    s.slice_chars(0, n - k)
                } else {
                    s.slice_chars(k, n)
                }));
            }
            Ok(Value::Str(s.clone()))
        }
        "center" | "ljust" | "rjust" => {
            let width = opt_int(a0)?
                .ok_or_else(|| type_error(format!("{name}() takes at least 1 argument")))?;
            let fill = match args.pos.get(1) {
                Some(v) => arg_sym(v, "fill")?.clone(),
                None => Sym::template(" "),
            };
            if fill.len_chars() != 1 {
                return Err(type_error(
                    "The fill character must be exactly one character long",
                ));
            }
            let n = s.len_chars() as i64;
            let marg = (width - n).max(0) as usize;
            let (l, r) = match name {
                "ljust" => (0, marg),
                "rjust" => (marg, 0),
                _ => {
                    let left = marg / 2 + (marg & width as usize & 1);
                    (left, marg - left)
                }
            };
            Ok(Value::Str(s.pad(l, r, &fill)))
        }
        "zfill" => {
            let width = opt_int(a0)?.unwrap_or(0);
            let n = s.len_chars() as i64;
            if width <= n {
                return Ok(Value::Str(s.clone()));
            }
            let zeros = Sym::template(&"0".repeat((width - n) as usize));
            if text.starts_with(['+', '-']) {
                let sign = s.slice_chars(0, 1);
                return Ok(Value::Str(
                    sign.concat(&zeros).concat(&s.slice_chars(1, n as usize)),
                ));
            }
            Ok(Value::Str(zeros.concat(s)))
        }
        "format" => format::str_format(s, &args.pos, &args.kw).map(Value::Str),
        "format_map" => {
            let mapping =
                a0.ok_or_else(|| type_error("format_map() takes exactly one argument"))?;
            let kw: Vec<(Rc<str>, Value)> = match mapping {
                Value::Dict(d) => {
                    d.iter().map(|(k, v)| (Rc::from(k.to_sym().as_str()), v.clone())).collect()
                }
                _ => return Err(type_error("format_map() argument must be a mapping")),
            };
            format::str_format(s, &[], &kw).map(Value::Str)
        }
        "isdigit" | "isdecimal" | "isnumeric" => {
            bool_v(!text.is_empty() && text.chars().all(|c| c.is_numeric()))
        }
        "isalpha" => bool_v(!text.is_empty() && text.chars().all(char::is_alphabetic)),
        "isalnum" => bool_v(!text.is_empty() && text.chars().all(char::is_alphanumeric)),
        "isspace" => bool_v(!text.is_empty() && text.chars().all(vllm_parser::python::py_isspace)),
        "islower" => {
            bool_v(text.chars().any(char::is_lowercase) && !text.chars().any(char::is_uppercase))
        }
        "isupper" => {
            bool_v(text.chars().any(char::is_uppercase) && !text.chars().any(char::is_lowercase))
        }
        "istitle" => bool_v(text.chars().any(|c| c.is_alphabetic()) && s.title().as_str() == text),
        "isascii" => bool_v(text.is_ascii()),
        "isprintable" => bool_v(Value::Str(s.clone()).repr().as_str().len() == text.len() + 2),
        "isidentifier" => bool_v(
            text.chars().next().is_some_and(|c| c == '_' || c.is_alphabetic())
                && text.chars().all(|c| c == '_' || c.is_alphanumeric()),
        ),
        _ => Err(type_error(format!(
            "'str' object has no attribute '{name}'"
        ))),
    }
}

/// Call a global function.
pub(crate) fn call_global(
    interp: &Interp<'_>,
    name: &str,
    args: Args,
    caller: Option<Value>,
) -> VResult<Value> {
    match name {
        "range" => {
            let ints = args
                .pos
                .iter()
                .map(|v| v.as_int().ok_or_else(|| type_error("range() integer expected")))
                .collect::<VResult<Vec<_>>>()?;
            let (start, stop, step) = match ints.as_slice() {
                [stop] => (0, *stop, 1),
                [start, stop] => (*start, *stop, 1),
                [start, stop, step] => (*start, *stop, *step),
                _ => return Err(type_error("range expected 1 to 3 arguments")),
            };
            if step == 0 {
                return Err(type_error("range() arg 3 must not be zero"));
            }
            let mut out = Vec::new();
            let mut i = start;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                out.push(Value::Int(i));
                i += step;
                if out.len() > 100_000 {
                    return Err(type_error("range too big"));
                }
            }
            Ok(Value::list(out))
        }
        "dict" | "namespace" => {
            let mut d = Dict::new();
            for p in &args.pos {
                match p {
                    Value::Dict(src) => {
                        for (k, v) in src.iter() {
                            d.insert(k.clone(), v.clone())?;
                        }
                    }
                    Value::Namespace(src) => {
                        for (k, v) in src.borrow().iter() {
                            d.insert(k.clone(), v.clone())?;
                        }
                    }
                    other => {
                        for item in other.iterate()? {
                            let pair = item.iterate()?;
                            if pair.len() != 2 {
                                return Err(type_error(
                                    "dictionary update sequence element has wrong length",
                                ));
                            }
                            d.insert(pair[0].clone(), pair[1].clone())?;
                        }
                    }
                }
            }
            for (k, v) in args.kw {
                d.insert(Value::template_str(&k), v)?;
            }
            Ok(if name == "dict" {
                Value::dict(d)
            } else {
                Value::Namespace(Rc::new(std::cell::RefCell::new(d)))
            })
        }
        "cycler" => {
            let mut helpers = interp.helpers.borrow_mut();
            helpers.push(Helper::Cycler {
                items: args.pos,
                pos: 0,
            });
            Ok(Value::Callable(Rc::new(Callable::Helper(
                helpers.len() - 1,
            ))))
        }
        "joiner" => {
            let sep = args.get(0, "sep").cloned().unwrap_or_else(|| Value::template_str(", "));
            let mut helpers = interp.helpers.borrow_mut();
            helpers.push(Helper::Joiner { sep, used: false });
            Ok(Value::Callable(Rc::new(Callable::Helper(
                helpers.len() - 1,
            ))))
        }
        "raise_exception" => {
            let message =
                args.pos.first().map(|v| v.to_sym().as_str().to_string()).unwrap_or_default();
            Err(type_error(message))
        }
        "strftime_now" => {
            let fmt = args.pos.first().map(|v| v.to_sym().as_str().to_string()).unwrap_or_default();
            let now = chrono::Local::now().format(&fmt).to_string();
            Ok(Value::Str(Sym::unknown(&now)))
        }
        "__hf_generation" => match caller {
            // transformers' {% generation %}: its body, as rendered without an assistant-token mask
            Some(c) => interp.call(&c, Args::default()),
            None => Ok(Value::Str(Sym::empty())),
        },
        _ => Err(type_error(format!("'{name}' is undefined"))),
    }
}

/// Call a `cycler` / `joiner` helper (`next()` is not modeled; calling a joiner gives its
/// separator after the first call).
pub(crate) fn call_helper(interp: &Interp<'_>, id: usize, _args: Args) -> VResult<Value> {
    let mut helpers = interp.helpers.borrow_mut();
    match &mut helpers[id] {
        Helper::Joiner { sep, used } => {
            if *used {
                Ok(sep.clone())
            } else {
                *used = true;
                Ok(Value::Str(Sym::empty()))
            }
        }
        Helper::Cycler { items, pos } => {
            if items.is_empty() {
                return Ok(Value::None);
            }
            let v = items[*pos % items.len()].clone();
            *pos += 1;
            Ok(v)
        }
    }
}

/// `fmt % args`.
pub(crate) fn percent_format(fmt: &Sym, args: &Value) -> VResult<Value> {
    format::percent(fmt, args).map(Value::Str)
}

fn attr_getter(attribute: &Value) -> Vec<String> {
    match attribute {
        Value::Str(s) => s.as_str().split('.').map(str::to_owned).collect(),
        Value::Int(i) => vec![i.to_string()],
        _ => Vec::new(),
    }
}

/// jinja2's `make_attrgetter`: dotted attribute paths, integer parts as indices.
fn get_path(v: &Value, path: &[String]) -> VResult<Value> {
    let mut cur = v.clone();
    for part in path {
        cur = match part.parse::<i64>() {
            Ok(i) => super::eval::getitem(&cur, &Value::Int(i))?,
            Err(_) => {
                let attr = getattr(&cur, part)?;
                if attr.is_undefined() {
                    super::eval::getitem(&cur, &Value::template_str(part))?
                } else {
                    attr
                }
            }
        };
    }
    Ok(cur)
}

fn select_items(
    interp: &Interp<'_>,
    value: &Value,
    args: &Args,
    by_attr: bool,
    keep: bool,
) -> VResult<Value> {
    let items = value.iterate()?;
    let (path, rest) = if by_attr {
        let a = args.pos.first().ok_or_else(|| type_error("missing attribute"))?;
        (attr_getter(a), &args.pos[1..])
    } else {
        (Vec::new(), &args.pos[..])
    };
    let mut out = Vec::new();
    for item in items {
        let subject = if by_attr {
            get_path(&item, &path)?
        } else {
            item.clone()
        };
        let ok = match rest.first() {
            None => subject.truthy(),
            Some(t) => {
                let tname = t.to_sym().as_str().to_string();
                let targs = Args {
                    pos: rest[1..].to_vec(),
                    kw: Vec::new(),
                };
                test(interp, &tname, &subject, &targs)?
            }
        };
        if ok == keep {
            out.push(item);
        }
    }
    Ok(Value::list(out))
}

fn sort_key(v: &Value, path: &[String], case_sensitive: bool) -> VResult<Value> {
    let k = if path.is_empty() {
        v.clone()
    } else {
        get_path(v, path)?
    };
    Ok(match (&k, case_sensitive) {
        (Value::Str(s), false) => Value::Str(s.lower()),
        _ => k,
    })
}

fn sorted(items: Vec<Value>, keys: Vec<Value>, reverse: bool) -> VResult<Vec<Value>> {
    let mut idx: Vec<usize> = (0..items.len()).collect();
    let mut err = None;
    idx.sort_by(|&a, &b| {
        let (x, y) = (&keys[a], &keys[b]);
        match (x.py_lt(y), y.py_lt(x)) {
            (Ok(true), _) => std::cmp::Ordering::Less,
            (_, Ok(true)) => std::cmp::Ordering::Greater,
            (Err(e), _) | (_, Err(e)) => {
                err.get_or_insert(e);
                std::cmp::Ordering::Equal
            }
            _ => std::cmp::Ordering::Equal,
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    if reverse {
        idx.reverse();
        // Python's sort is stable for reverse=True too: equal items keep their order
        let mut out: Vec<usize> = Vec::with_capacity(idx.len());
        let mut i = 0;
        while i < idx.len() {
            let mut j = i + 1;
            while j < idx.len() && keys[idx[j]].py_eq(&keys[idx[i]]).unwrap_or(false) {
                j += 1;
            }
            out.extend(idx[i..j].iter().rev());
            i = j;
        }
        idx = out;
    }
    Ok(idx.into_iter().map(|i| items[i].clone()).collect())
}

/// Apply a filter.
pub(crate) fn filter(interp: &Interp<'_>, name: &str, value: Value, args: Args) -> VResult<Value> {
    let a0 = args.pos.first();
    match name {
        "default" | "d" => {
            let boolean = args.get(1, "boolean").is_some_and(Value::truthy);
            let missing = value.is_undefined() || (boolean && !value.truthy());
            Ok(if missing {
                args.get(0, "default_value")
                    .cloned()
                    .unwrap_or_else(|| Value::Str(Sym::empty()))
            } else {
                value
            })
        }
        "length" | "count" => Ok(Value::Int(value.py_len()? as i64)),
        "string" => Ok(Value::Str(value.to_sym())),
        "safe" | "e" | "escape" | "forceescape" => match name {
            "safe" => Ok(value),
            _ => Ok(Value::Str(escape(&value.to_sym()))),
        },
        "tojson" => {
            let indent = args.get(0, "indent").cloned();
            json::tojson(
                &value,
                args.kwarg("ensure_ascii").is_some_and(Value::truthy),
                indent.as_ref(),
                args.kwarg("separators"),
                args.kwarg("sort_keys").is_some_and(Value::truthy),
            )
            .map(Value::Str)
        }
        "trim" => {
            let s = value.to_sym();
            let chars = match a0 {
                None | Some(Value::None) => None,
                Some(v) => Some(arg_sym(v, "trim")?.clone()),
            };
            Ok(Value::Str(s.strip(chars.as_ref(), true, true)?))
        }
        "lower" => Ok(Value::Str(value.to_sym().lower())),
        "upper" => Ok(Value::Str(value.to_sym().upper())),
        "capitalize" => Ok(Value::Str(value.to_sym().capitalize())),
        "title" => Ok(Value::Str(jinja_title(&value.to_sym()))),
        "replace" => {
            let s = value.to_sym();
            let old = a0.ok_or_else(|| type_error("replace() missing old"))?.to_sym();
            let new = args.pos.get(1).ok_or_else(|| type_error("replace() missing new"))?.to_sym();
            let count = opt_int(args.get(2, "count"))?.unwrap_or(-1);
            Ok(Value::Str(s.replace(&old, &new, count)?))
        }
        "items" => match &value {
            Value::Undefined(_) => Ok(Value::list(Vec::new())),
            Value::Dict(d) => Ok(Value::list(
                d.iter().map(|(k, v)| Value::tuple(vec![k.clone(), v.clone()])).collect(),
            )),
            other => Err(type_error(format!(
                "Can only get item pairs from a mapping, not {}",
                other.type_name()
            ))),
        },
        "list" => Ok(Value::list(value.iterate()?)),
        "first" => {
            Ok(value.iterate()?.into_iter().next().unwrap_or_else(|| Value::undefined("first")))
        }
        "last" => Ok(value
            .iterate()?
            .into_iter()
            .next_back()
            .unwrap_or_else(|| Value::undefined("last"))),
        "reverse" => match &value {
            Value::Str(s) => Ok(Value::Str(s.slice(None, None, Some(-1))?)),
            other => {
                let mut items = other.iterate()?;
                items.reverse();
                Ok(Value::list(items))
            }
        },
        "join" => {
            let sep = match args.get(0, "d") {
                Some(v) => v.to_sym(),
                None => Sym::empty(),
            };
            let mut items = value.iterate()?;
            if let Some(attr) = args.get(1, "attribute") {
                let path = attr_getter(attr);
                items = items.iter().map(|x| get_path(x, &path)).collect::<VResult<_>>()?;
            }
            let parts: Vec<Sym> = items.iter().map(Value::to_sym).collect();
            Ok(Value::Str(sep.join(&parts)))
        }
        "map" => {
            let items = value.iterate()?;
            if let Some(attr) = args.kwarg("attribute") {
                let path = attr_getter(attr);
                let default = args.kwarg("default").cloned();
                return Ok(Value::list(
                    items
                        .iter()
                        .map(|x| {
                            let v = get_path(x, &path)?;
                            Ok(match (&v, &default) {
                                (Value::Undefined(_), Some(d)) => d.clone(),
                                _ => v,
                            })
                        })
                        .collect::<VResult<_>>()?,
                ));
            }
            let fname = a0.ok_or_else(|| type_error("map requires a filter argument"))?.to_sym();
            let rest = Args {
                pos: args.pos[1..].to_vec(),
                kw: args.kw.clone(),
            };
            Ok(Value::list(
                items
                    .into_iter()
                    .map(|x| {
                        filter(
                            interp,
                            fname.as_str(),
                            x,
                            Args {
                                pos: rest.pos.clone(),
                                kw: rest.kw.clone(),
                            },
                        )
                    })
                    .collect::<VResult<_>>()?,
            ))
        }
        "select" => select_items(interp, &value, &args, false, true),
        "reject" => select_items(interp, &value, &args, false, false),
        "selectattr" => select_items(interp, &value, &args, true, true),
        "rejectattr" => select_items(interp, &value, &args, true, false),
        "sort" => {
            let reverse = args.get(0, "reverse").is_some_and(Value::truthy);
            let case_sensitive = args.get(1, "case_sensitive").is_some_and(Value::truthy);
            let path = args.get(2, "attribute").map(attr_getter).unwrap_or_default();
            let items = value.iterate()?;
            let keys = items
                .iter()
                .map(|x| sort_key(x, &path, case_sensitive))
                .collect::<VResult<_>>()?;
            Ok(Value::list(sorted(items, keys, reverse)?))
        }
        "dictsort" => {
            let Value::Dict(d) = &value else {
                return Err(type_error("dictsort requires a mapping"));
            };
            let case_sensitive = args.get(0, "case_sensitive").is_some_and(Value::truthy);
            let by_value = args.get(1, "by").is_some_and(|v| v.to_sym().as_str() == "value");
            let reverse = args.get(2, "reverse").is_some_and(Value::truthy);
            let items: Vec<Value> =
                d.iter().map(|(k, v)| Value::tuple(vec![k.clone(), v.clone()])).collect();
            let keys = d
                .iter()
                .map(|(k, v)| sort_key(if by_value { v } else { k }, &[], case_sensitive))
                .collect::<VResult<_>>()?;
            Ok(Value::list(sorted(items, keys, reverse)?))
        }
        "unique" => {
            let case_sensitive = args.get(0, "case_sensitive").is_some_and(Value::truthy);
            let path = args.get(1, "attribute").map(attr_getter).unwrap_or_default();
            let mut seen: Vec<Value> = Vec::new();
            let mut out = Vec::new();
            for item in value.iterate()? {
                let k = sort_key(&item, &path, case_sensitive)?;
                let mut dup = false;
                for s in &seen {
                    if s.py_eq(&k)? {
                        dup = true;
                        break;
                    }
                }
                if !dup {
                    seen.push(k);
                    out.push(item);
                }
            }
            Ok(Value::list(out))
        }
        "int" => Ok(match &value {
            Value::Int(_) => value,
            Value::Bool(b) => Value::Int(i64::from(*b)),
            Value::Float(f) => Value::Int(f.trunc() as i64),
            Value::Str(s) => Value::Int(
                s.as_str()
                    .trim()
                    .parse::<i64>()
                    .or_else(|_| s.as_str().trim().parse::<f64>().map(|f| f as i64))
                    .unwrap_or_else(|_| {
                        args.get(0, "default").and_then(Value::as_int).unwrap_or(0)
                    }),
            ),
            _ => args.get(0, "default").cloned().unwrap_or(Value::Int(0)),
        }),
        "float" => Ok(match &value {
            Value::Float(_) => value,
            v if v.as_number().is_some() => Value::Float(v.as_number().expect("checked")),
            Value::Str(s) => Value::Float(s.as_str().trim().parse::<f64>().unwrap_or(0.0)),
            _ => Value::Float(0.0),
        }),
        "abs" => Ok(match &value {
            Value::Int(i) => Value::Int(i.abs()),
            Value::Float(f) => Value::Float(f.abs()),
            Value::Bool(b) => Value::Int(i64::from(*b)),
            other => {
                return Err(type_error(format!(
                    "bad operand type for abs(): '{}'",
                    other.type_name()
                )));
            }
        }),
        "round" => {
            let precision = opt_int(args.get(0, "precision"))?.unwrap_or(0);
            let method = args
                .get(1, "method")
                .map(|v| v.to_sym().as_str().to_string())
                .unwrap_or_else(|| "common".into());
            let x = value.as_number().ok_or_else(|| type_error("round requires a number"))?;
            let m = 10f64.powi(precision as i32);
            let r = match method.as_str() {
                "ceil" => (x * m).ceil() / m,
                "floor" => (x * m).floor() / m,
                _ => (x * m).round() / m,
            };
            Ok(Value::Float(r))
        }
        "sum" => {
            let mut total = args.get(1, "start").cloned().unwrap_or(Value::Int(0));
            let path = args.get(0, "attribute").map(attr_getter).unwrap_or_default();
            for item in value.iterate()? {
                let v = if path.is_empty() {
                    item
                } else {
                    get_path(&item, &path)?
                };
                total = super::eval::binop(super::ir::BinOp::Add, &total, &v)?;
            }
            Ok(total)
        }
        "min" | "max" => {
            let items = value.iterate()?;
            let case_sensitive = args.get(0, "case_sensitive").is_some_and(Value::truthy);
            let path = args.get(1, "attribute").map(attr_getter).unwrap_or_default();
            let mut best: Option<(Value, Value)> = None;
            for item in items {
                let k = sort_key(&item, &path, case_sensitive)?;
                let better = match &best {
                    None => true,
                    Some((bk, _)) => {
                        if name == "min" {
                            k.py_lt(bk)?
                        } else {
                            bk.py_lt(&k)?
                        }
                    }
                };
                if better {
                    best = Some((k, item));
                }
            }
            Ok(best.map(|(_, v)| v).unwrap_or_else(|| Value::undefined(name)))
        }
        "attr" => {
            let n = a0.ok_or_else(|| type_error("attr requires a name"))?.to_sym();
            getattr(&value, n.as_str())
        }
        "batch" | "slice" => {
            let n = opt_int(a0)?.ok_or_else(|| type_error("missing size"))?.max(1) as usize;
            let items = value.iterate()?;
            let fill = args.pos.get(1).cloned();
            let mut out = Vec::new();
            if name == "batch" {
                for chunk in items.chunks(n) {
                    let mut c = chunk.to_vec();
                    if let Some(f) = &fill {
                        while c.len() < n {
                            c.push(f.clone());
                        }
                    }
                    out.push(Value::list(c));
                }
            } else {
                let per = items.len() / n;
                let extra = items.len() % n;
                let mut offset = 0;
                for i in 0..n {
                    let start = offset + i * per + i.min(extra);
                    let end = start + per + usize::from(i < extra);
                    let mut c = items[start.min(items.len())..end.min(items.len())].to_vec();
                    if let Some(f) = &fill
                        && i >= extra
                        && extra > 0
                    {
                        c.push(f.clone());
                    }
                    out.push(Value::list(c));
                    offset = 0;
                }
            }
            Ok(Value::list(out))
        }
        "center" => {
            let width = opt_int(a0)?.unwrap_or(80);
            str_method(
                &value.to_sym(),
                "center",
                &Args {
                    pos: vec![Value::Int(width)],
                    kw: Vec::new(),
                },
            )
        }
        "indent" => indent(&value.to_sym(), &args),
        "truncate" => {
            let s = value.to_sym();
            let length = opt_int(args.get(0, "length"))?.unwrap_or(255) as usize;
            let killwords = args.get(1, "killwords").is_some_and(Value::truthy);
            let end = args.get(2, "end").map(Value::to_sym).unwrap_or_else(|| Sym::template("..."));
            let leeway = opt_int(args.get(3, "leeway"))?.unwrap_or(5) as usize;
            let n = s.len_chars();
            if n <= length + leeway {
                return Ok(Value::Str(s));
            }
            let keep = length.saturating_sub(end.len_chars());
            if killwords {
                return Ok(Value::Str(s.slice_chars(0, keep).concat(&end)));
            }
            let head = s.slice_chars(0, keep);
            let cut = head.as_str().rfind(' ').map_or(keep, |b| head.as_str()[..b].chars().count());
            Ok(Value::Str(s.slice_chars(0, cut).concat(&end)))
        }
        "wordcount" => Ok(Value::Int(
            value.to_sym().as_str().split_whitespace().count() as i64,
        )),
        "format" => {
            let s = value.to_sym();
            if !args.kw.is_empty() {
                let d = Value::dict({
                    let mut d = Dict::new();
                    for (k, v) in &args.kw {
                        d.insert(Value::template_str(k), v.clone())?;
                    }
                    d
                });
                return format::percent(&s, &d).map(Value::Str);
            }
            format::percent(&s, &Value::tuple(args.pos.clone())).map(Value::Str)
        }
        _ => Err(type_error(format!("no filter named '{name}'"))),
    }
}

/// jinja2's `title` filter: words split at `-`, whitespace and opening brackets.
fn jinja_title(s: &Sym) -> Sym {
    let text = s.as_str();
    let mut b = SymBuilder::new();
    let mut pos = 0;
    let mut word_start = true;
    for (i, c) in text.char_indices() {
        if matches!(c, '-' | '(' | '{' | '[' | '<') || vllm_parser::python::py_isspace(c) {
            if pos < i {
                push_titled(&mut b, s, pos, i, word_start);
            }
            push_slice(&mut b, s, i, i + c.len_utf8());
            pos = i + c.len_utf8();
            word_start = true;
        }
    }
    if pos < text.len() {
        push_titled(&mut b, s, pos, text.len(), word_start);
    }
    let _ = word_start;
    b.finish()
}

fn push_slice(b: &mut SymBuilder, s: &Sym, a: usize, e: usize) {
    b.push_slice(s, a, e);
}

/// item[0].upper() + item[1:].lower()
fn push_titled(b: &mut SymBuilder, s: &Sym, a: usize, e: usize, _start: bool) {
    let word = {
        let mut w = SymBuilder::new();
        w.push_slice(s, a, e);
        w.finish()
    };
    let n = word.len_chars();
    b.push_sym(&word.slice_chars(0, 1).upper());
    b.push_sym(&word.slice_chars(1, n).lower());
}

/// markupsafe's `escape`: a control token keeps its id if its spelling needs no escaping.
fn escape(s: &Sym) -> Sym {
    let mut b = SymBuilder::new();
    for (kind, text) in s.runs() {
        let mut esc = String::with_capacity(text.len());
        for c in text.chars() {
            match c {
                '&' => esc.push_str("&amp;"),
                '<' => esc.push_str("&lt;"),
                '>' => esc.push_str("&gt;"),
                '"' => esc.push_str("&#34;"),
                '\'' => esc.push_str("&#39;"),
                c => esc.push(c),
            }
        }
        let kind = if esc == text {
            kind
        } else if matches!(kind, Kind::Control(_)) {
            Kind::Template
        } else {
            kind
        };
        b.push(kind, &esc);
    }
    b.finish()
}

/// jinja2's `indent(width=4, first=False, blank=False)`.
fn indent(s: &Sym, args: &Args) -> VResult<Value> {
    let width = args.get(0, "width");
    let indention = match width {
        Some(Value::Str(w)) => w.clone(),
        Some(v) => Sym::template(&" ".repeat(v.as_int().unwrap_or(4).max(0) as usize)),
        None => Sym::template("    "),
    };
    let first = args.get(1, "first").is_some_and(Value::truthy);
    let blank = args.get(2, "blank").is_some_and(Value::truthy);
    let newline = Sym::template("\n");
    let s = s.concat(&newline); // jinja2's own quirk (for splitlines)
    let lines = s.splitlines(false);
    let mut rv = if blank {
        newline.concat(&indention).join(&lines)
    } else {
        let mut it = lines.into_iter();
        let mut rv = it.next().unwrap_or_else(Sym::empty);
        let rest: Vec<Sym> = it
            .map(|line| {
                if line.is_empty() {
                    line
                } else {
                    indention.concat(&line)
                }
            })
            .collect();
        if !rest.is_empty() {
            rv = rv.concat(&newline).concat(&newline.join(&rest));
        }
        rv
    };
    if first {
        rv = indention.concat(&rv);
    }
    Ok(Value::Str(rv))
}

/// Apply a test.
pub(crate) fn test(interp: &Interp<'_>, name: &str, v: &Value, args: &Args) -> VResult<bool> {
    let a0 = args.pos.first();
    let other = || a0.cloned().ok_or_else(|| type_error(format!("test {name} needs an argument")));
    Ok(match name {
        "defined" => !v.is_undefined(),
        "undefined" => v.is_undefined(),
        "none" => matches!(v, Value::None),
        "string" => matches!(v, Value::Str(_)),
        "number" => matches!(v, Value::Int(_) | Value::Float(_) | Value::Bool(_)),
        "integer" => matches!(v, Value::Int(_)),
        "float" => matches!(v, Value::Float(_)),
        "boolean" => matches!(v, Value::Bool(_)),
        "true" => matches!(v, Value::Bool(true)),
        "false" => matches!(v, Value::Bool(false)),
        "mapping" => matches!(v, Value::Dict(_)),
        "iterable" => matches!(
            v,
            Value::Str(_) | Value::List(_) | Value::Tuple(_) | Value::Dict(_) | Value::Undefined(_)
        ),
        "sequence" => matches!(
            v,
            Value::Str(_) | Value::List(_) | Value::Tuple(_) | Value::Dict(_)
        ),
        "callable" => matches!(v, Value::Callable(_)),
        "escaped" => false,
        "lower" => {
            let t = v.to_sym();
            t.as_str().chars().any(char::is_lowercase)
                && !t.as_str().chars().any(char::is_uppercase)
        }
        "upper" => {
            let t = v.to_sym();
            t.as_str().chars().any(char::is_uppercase)
                && !t.as_str().chars().any(char::is_lowercase)
        }
        "odd" => v.as_int().ok_or_else(|| type_error("odd requires an integer"))? % 2 != 0,
        "even" => v.as_int().ok_or_else(|| type_error("even requires an integer"))? % 2 == 0,
        "divisibleby" => {
            let d =
                other()?.as_int().ok_or_else(|| type_error("divisibleby requires an integer"))?;
            d != 0 && v.as_int().is_some_and(|x| x % d == 0)
        }
        "eq" | "equalto" | "==" => v.py_eq(&other()?)?,
        "ne" | "!=" => !v.py_eq(&other()?)?,
        "lt" | "lessthan" | "<" => compare(CmpOp::Lt, v, &other()?)?,
        "le" | "<=" => compare(CmpOp::Lte, v, &other()?)?,
        "gt" | "greaterthan" | ">" => compare(CmpOp::Gt, v, &other()?)?,
        "ge" | ">=" => compare(CmpOp::Gte, v, &other()?)?,
        "in" => contains(&other()?, v)?,
        "sameas" => match (v, &other()?) {
            (Value::None, Value::None) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Dict(a), Value::Dict(b)) => Rc::ptr_eq(a, b),
            (Value::List(a), Value::List(b)) => Rc::ptr_eq(a, b),
            _ => false,
        },
        _ => {
            let _ = interp;
            return Err(type_error(format!("no test named '{name}'")));
        }
    })
}
