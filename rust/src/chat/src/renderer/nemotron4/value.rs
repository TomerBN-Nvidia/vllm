// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Template values with Python's semantics (jinja2 runs chat templates on Python objects): every
//! string is a [`Sym`], containers and numbers compare, convert and print as Python's do.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use indexmap::IndexMap;
use vllm_parser::python::float_repr;

use super::sym::{Kind, Sym, SymBuilder, SymError};

/// A template value.
#[derive(Clone)]
pub(crate) enum Value {
    /// jinja2's `Undefined`, with the name that was looked up (or a description).
    Undefined(Rc<str>),
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Sym),
    List(Rc<Vec<Value>>),
    Tuple(Rc<Vec<Value>>),
    Dict(Rc<Dict>),
    /// jinja2's `namespace()`: attributes assigned with `{% set ns.x = ... %}`.
    Namespace(Rc<RefCell<Dict>>),
    /// Macros, global functions, bound methods, `caller`, loop helpers.
    Callable(Rc<Callable>),
    /// The `loop` variable of a for loop.
    Loop(Rc<LoopState>),
}

/// A callable value; the interpreter calls it.
pub(crate) enum Callable {
    Macro(Rc<super::eval::MacroValue>),
    Builtin(&'static str),
    /// `receiver.name(...)`.
    Method(Value, Rc<str>),
    /// A `cycler(...)` / `joiner(...)` object's state lives in the interpreter: their id.
    Helper(usize),
}

/// The state of one loop iteration, as jinja2's `LoopContext` shows it.
pub(crate) struct LoopState {
    pub(crate) index0: usize,
    pub(crate) length: usize,
    pub(crate) depth0: usize,
    pub(crate) previtem: Option<Value>,
    pub(crate) nextitem: Option<Value>,
    pub(crate) changed_last: RefCell<Option<Vec<Value>>>,
}

/// A dict key: Python's hash-equality (`1 == 1.0 == True`; strings by their text).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Key {
    Str(Rc<str>),
    Int(i64),
    Float(u64),
    None,
}

/// A Python dict: insertion order, each key with its original value.
#[derive(Clone, Default)]
pub(crate) struct Dict {
    map: IndexMap<Key, (Value, Value)>,
}

impl Dict {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn insert(&mut self, key: Value, value: Value) -> Result<(), SymError> {
        let k = key.to_key()?;
        if let Some(slot) = self.map.get_mut(&k) {
            slot.1 = value; // a repeated key keeps its first position
        } else {
            self.map.insert(k, (key, value));
        }
        Ok(())
    }

    pub(crate) fn get(&self, key: &Value) -> Option<&Value> {
        key.to_key().ok().and_then(|k| self.map.get(&k)).map(|(_, v)| v)
    }

    pub(crate) fn get_str(&self, key: &str) -> Option<&Value> {
        self.map.get(&Key::Str(Rc::from(key))).map(|(_, v)| v)
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&Value, &Value)> {
        self.map.values().map(|(k, v)| (k, v))
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &Value> {
        self.map.values().map(|(k, _)| k)
    }
}

/// A Python error in a template operation (raised as the template's error).
pub(crate) type VResult<T> = Result<T, SymError>;

pub(crate) fn type_error(message: impl Into<String>) -> SymError {
    SymError::Python(message.into())
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Undefined(name) => write!(f, "Undefined({name})"),
            Value::Str(s) => write!(f, "{s:?}"),
            other => f.write_str(other.repr().as_str()),
        }
    }
}

impl Value {
    pub(crate) fn template_str(s: &str) -> Value {
        Value::Str(Sym::template(s))
    }

    pub(crate) fn list(items: Vec<Value>) -> Value {
        Value::List(Rc::new(items))
    }

    pub(crate) fn tuple(items: Vec<Value>) -> Value {
        Value::Tuple(Rc::new(items))
    }

    pub(crate) fn dict(d: Dict) -> Value {
        Value::Dict(Rc::new(d))
    }

    pub(crate) fn undefined(name: &str) -> Value {
        Value::Undefined(Rc::from(name))
    }

    pub(crate) fn is_undefined(&self) -> bool {
        matches!(self, Value::Undefined(_))
    }

    /// Python's type name.
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Value::Undefined(_) => "Undefined",
            Value::None => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Tuple(_) => "tuple",
            Value::Dict(_) => "dict",
            Value::Namespace(_) => "Namespace",
            Value::Callable(_) => "function",
            Value::Loop(_) => "LoopContext",
        }
    }

    pub(crate) fn as_sym(&self) -> Option<&Sym> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Python's truth value.
    pub(crate) fn truthy(&self) -> bool {
        match self {
            Value::Undefined(_) | Value::None => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(items) | Value::Tuple(items) => !items.is_empty(),
            Value::Dict(d) => !d.is_empty(),
            Value::Namespace(_) | Value::Callable(_) | Value::Loop(_) => true,
        }
    }

    /// The dict key of this value.
    pub(crate) fn to_key(&self) -> VResult<Key> {
        Ok(match self {
            Value::Str(s) => Key::Str(Rc::from(s.as_str())),
            Value::Bool(b) => Key::Int(i64::from(*b)),
            Value::Int(i) => Key::Int(*i),
            Value::Float(f) if f.fract() == 0.0 && f.abs() < 9.0e15 => Key::Int(*f as i64),
            Value::Float(f) => Key::Float(f.to_bits()),
            Value::None => Key::None,
            Value::Tuple(_) => Key::Str(Rc::from(self.repr().as_str())),
            other => {
                return Err(type_error(format!(
                    "unhashable type: '{}'",
                    other.type_name()
                )));
            }
        })
    }

    /// A number as f64 (bools are ints), for arithmetic and comparison.
    pub(crate) fn as_number(&self) -> Option<f64> {
        match self {
            Value::Bool(b) => Some(f64::from(u8::from(*b))),
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub(crate) fn as_int(&self) -> Option<i64> {
        match self {
            Value::Bool(b) => Some(i64::from(*b)),
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Python's `==`.
    pub(crate) fn py_eq(&self, other: &Value) -> VResult<bool> {
        Ok(match (self, other) {
            (Value::Undefined(_), Value::Undefined(_)) => true,
            (Value::Undefined(_), _) | (_, Value::Undefined(_)) => false,
            (Value::None, Value::None) => true,
            (Value::Str(a), Value::Str(b)) => a.py_eq(b)?,
            (Value::Int(a), Value::Int(b)) => a == b,
            (a, b) if a.as_number().is_some() && b.as_number().is_some() => {
                match (a.as_int(), b.as_int()) {
                    (Some(x), Some(y)) => x == y,
                    _ => a.as_number() == b.as_number(),
                }
            }
            (Value::List(a), Value::List(b)) | (Value::Tuple(a), Value::Tuple(b)) => {
                if a.len() != b.len() {
                    return Ok(false);
                }
                for (x, y) in a.iter().zip(b.iter()) {
                    if !x.py_eq(y)? {
                        return Ok(false);
                    }
                }
                true
            }
            (Value::Dict(a), Value::Dict(b)) => {
                if a.len() != b.len() {
                    return Ok(false);
                }
                for (k, v) in a.iter() {
                    match b.get(k) {
                        Some(w) if v.py_eq(w)? => {}
                        _ => return Ok(false),
                    }
                }
                true
            }
            (Value::Namespace(a), Value::Namespace(b)) => Rc::ptr_eq(a, b),
            _ => false,
        })
    }

    /// Python's `<` (and, through it, `<=`, `>`, `>=`).
    pub(crate) fn py_lt(&self, other: &Value) -> VResult<bool> {
        Ok(match (self, other) {
            (Value::Str(a), Value::Str(b)) => a.as_str() < b.as_str(),
            (a, b) if a.as_number().is_some() && b.as_number().is_some() => {
                match (a.as_int(), b.as_int()) {
                    (Some(x), Some(y)) => x < y,
                    _ => a.as_number() < b.as_number(),
                }
            }
            (Value::List(a), Value::List(b)) | (Value::Tuple(a), Value::Tuple(b)) => {
                for (x, y) in a.iter().zip(b.iter()) {
                    if !x.py_eq(y)? {
                        return x.py_lt(y);
                    }
                }
                a.len() < b.len()
            }
            (a, b) => {
                return Err(type_error(format!(
                    "'<' not supported between instances of '{}' and '{}'",
                    a.type_name(),
                    b.type_name()
                )));
            }
        })
    }

    /// Python's `str(x)`, with runs: a string itself, numbers and the like as template text,
    /// containers by their repr, `Undefined` empty.
    pub(crate) fn to_sym(&self) -> Sym {
        match self {
            Value::Str(s) => s.clone(),
            Value::Undefined(_) => Sym::empty(),
            Value::None => Sym::template("None"),
            Value::Bool(b) => Sym::template(if *b { "True" } else { "False" }),
            Value::Int(i) => Sym::template(&i.to_string()),
            Value::Float(f) => Sym::template(&float_repr(*f)),
            Value::List(_) | Value::Tuple(_) | Value::Dict(_) => self.repr(),
            Value::Namespace(ns) => {
                let mut b = SymBuilder::new();
                b.push(Kind::Template, "<Namespace ");
                b.push_sym(&Value::Dict(Rc::new(ns.borrow().clone())).repr());
                b.push(Kind::Template, ">");
                b.finish()
            }
            Value::Callable(_) => Sym::unknown("<function>"),
            Value::Loop(_) => Sym::unknown("<LoopContext>"),
        }
    }

    /// Python's `repr(x)`, with runs (a string's own runs kept between its quotes).
    pub(crate) fn repr(&self) -> Sym {
        let mut b = SymBuilder::new();
        self.repr_into(&mut b);
        b.finish()
    }

    fn repr_into(&self, b: &mut SymBuilder) {
        match self {
            Value::Str(s) => repr_str_into(s, b),
            Value::List(items) => {
                b.push(Kind::Template, "[");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        b.push(Kind::Template, ", ");
                    }
                    item.repr_into(b);
                }
                b.push(Kind::Template, "]");
            }
            Value::Tuple(items) => {
                b.push(Kind::Template, "(");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        b.push(Kind::Template, ", ");
                    }
                    item.repr_into(b);
                }
                if items.len() == 1 {
                    b.push(Kind::Template, ",");
                }
                b.push(Kind::Template, ")");
            }
            Value::Dict(d) => {
                b.push(Kind::Template, "{");
                for (i, (k, v)) in d.iter().enumerate() {
                    if i > 0 {
                        b.push(Kind::Template, ", ");
                    }
                    k.repr_into(b);
                    b.push(Kind::Template, ": ");
                    v.repr_into(b);
                }
                b.push(Kind::Template, "}");
            }
            Value::Undefined(_) => b.push(Kind::Unknown, "Undefined"),
            other => b.push_sym(&other.to_sym()),
        }
    }

    /// The items a `for` loop or `list()` sees.
    pub(crate) fn iterate(&self) -> VResult<Vec<Value>> {
        Ok(match self {
            Value::List(items) | Value::Tuple(items) => items.as_ref().clone(),
            Value::Dict(d) => d.keys().cloned().collect(),
            Value::Str(s) => s.chars().into_iter().map(Value::Str).collect(),
            Value::Undefined(_) => Vec::new(),
            other => {
                return Err(type_error(format!(
                    "'{}' object is not iterable",
                    other.type_name()
                )));
            }
        })
    }

    /// Python's `len`.
    pub(crate) fn py_len(&self) -> VResult<usize> {
        Ok(match self {
            Value::Str(s) => s.len_chars(),
            Value::List(items) | Value::Tuple(items) => items.len(),
            Value::Dict(d) => d.len(),
            Value::Undefined(_) => 0,
            other => {
                return Err(type_error(format!(
                    "object of type '{}' has no len()",
                    other.type_name()
                )));
            }
        })
    }
}

/// Python's repr of a string: quotes (single, or double when the text has a single quote and no
/// double one), escapes; a control token keeps its id when its spelling needs no escaping.
pub(crate) fn repr_str_into(s: &Sym, b: &mut SymBuilder) {
    let v = s.as_str();
    let q = if v.contains('\'') && !v.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut quote = [0u8; 4];
    let quote = q.encode_utf8(&mut quote);
    b.push(Kind::Template, quote);
    for (kind, text) in s.runs() {
        let esc = repr_escape(text, q);
        let kind = if esc == text { kind } else { kind_cut(kind) };
        b.push(kind, &esc);
    }
    b.push(Kind::Template, quote);
}

fn kind_cut(kind: Kind) -> Kind {
    match kind {
        Kind::Control(_) => Kind::Template,
        other => other,
    }
}

/// `repr()`'s escaping of text inside quotes `q`.
fn repr_escape(text: &str, q: char) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if is_printable(c) => out.push(c),
            c if (c as u32) < 0x100 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) < 0x10000 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out
}

/// Python's `str.isprintable` for one character (approximately: controls, separators other than
/// space, and unassigned private-use are not printable).
fn is_printable(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    !(c.is_control()
        || c.is_whitespace()
        || matches!(c as u32, 0xad | 0x600..=0x605 | 0x200b..=0x200f | 0x2028..=0x202e | 0x2060..=0x2064 | 0xfeff)
        || (0xe000..=0xf8ff).contains(&(c as u32))
        || (0xf0000..=0x10ffff).contains(&(c as u32)))
}
