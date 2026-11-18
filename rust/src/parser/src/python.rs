// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python's value semantics where the nemotron4 parser's and renderer's results depend on them.
//!
//! The Python plugins build tool-call arguments with `json.dumps(..., ensure_ascii=False)`, coerce
//! raw strings with `int()`, `float()`, `json.loads()` and `ast.literal_eval()`, and strip with
//! `str.strip()`. The Rust port gives the same strings: integers keep every digit, floats print as
//! Python's `repr`, objects keep their insertion order.

use std::fmt::Write as _;

/// A JSON-able Python value.
#[derive(Debug, Clone, PartialEq)]
pub enum PyVal {
    Null,
    Bool(bool),
    /// An integer of any size, as its canonical decimal digits (an optional `-`, no leading zeros).
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<PyVal>),
    /// A dict in insertion order; a repeated key keeps its first position and its last value.
    Dict(Vec<(String, PyVal)>),
}

impl PyVal {
    pub(crate) fn dict_insert(items: &mut Vec<(String, PyVal)>, key: String, value: PyVal) {
        if let Some(slot) = items.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            items.push((key, value));
        }
    }

    fn is_finite(&self) -> bool {
        match self {
            PyVal::Float(f) => f.is_finite(),
            PyVal::List(items) => items.iter().all(PyVal::is_finite),
            PyVal::Dict(items) => items.iter().all(|(_, v)| v.is_finite()),
            _ => true,
        }
    }
}

/// `str.isspace()` for one character: Unicode White_Space plus the information separators
/// U+001C..U+001F.
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// `str.lstrip()`.
pub fn py_lstrip(s: &str) -> &str {
    s.trim_start_matches(py_isspace)
}

/// `json.dumps(value, ensure_ascii=False)`.
pub fn dumps(value: &PyVal) -> String {
    let mut out = String::new();
    dump_into(value, &mut out);
    out
}

fn dump_into(value: &PyVal, out: &mut String) {
    match value {
        PyVal::Null => out.push_str("null"),
        PyVal::Bool(true) => out.push_str("true"),
        PyVal::Bool(false) => out.push_str("false"),
        PyVal::Int(digits) => out.push_str(digits),
        PyVal::Float(f) => out.push_str(&float_repr(*f)),
        PyVal::Str(s) => dump_str(s, out),
        PyVal::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_into(item, out);
            }
            out.push(']');
        }
        PyVal::Dict(items) => {
            out.push('{');
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_str(k, out);
                out.push_str(": ");
                dump_into(v, out);
            }
            out.push('}');
        }
    }
}

/// A JSON string as `json.dumps(s, ensure_ascii=False)` writes it.
pub fn dump_str(s: &str, out: &mut String) {
    out.push('"');
    escape_str_into(s, out);
    out.push('"');
}

/// The inside of `dump_str`'s quotes.
pub fn escape_str_into(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

/// `repr(f)` for a finite float, as `json.dumps` writes it (shortest round-trip digits; scientific
/// notation for exponents below -4 or from 16 up, with a sign and at least two exponent digits).
pub fn float_repr(f: f64) -> String {
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    if !f.is_finite() {
        return if f.is_nan() {
            "NaN".into()
        } else if f > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    let sci = format!("{:e}", f.abs()); // shortest digits: "1.2345e2", "1e-5"
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if f.is_sign_negative() { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let s = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{s}")
    } else {
        let mant = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mant}e{esign}{:02}", exp.abs())
    }
}

/// The exact integer value of a finite, integral float, as decimal digits (`int(f)`).
pub fn float_to_int_digits(f: f64) -> String {
    let s = format!("{f:.0}");
    normalize_int_digits(&s)
}

fn normalize_int_digits(s: &str) -> String {
    let (neg, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        "0".into()
    } else if neg {
        format!("-{digits}")
    } else {
        digits.to_string()
    }
}

/// `int(s)` for a string: whitespace around, an optional sign, ASCII digits with single underscores
/// between them.
pub fn py_int(s: &str) -> Option<String> {
    let t = py_strip(s);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if !digits_with_underscores(body) {
        return None;
    }
    let digits: String = body.chars().filter(|c| *c != '_').collect();
    Some(normalize_int_digits(&if neg {
        format!("-{digits}")
    } else {
        digits
    }))
}

fn digits_with_underscores(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || !b[0].is_ascii_digit() || !b[b.len() - 1].is_ascii_digit() {
        return false;
    }
    let mut prev_underscore = false;
    for &c in b {
        if c == b'_' {
            if prev_underscore {
                return false;
            }
            prev_underscore = true;
        } else if c.is_ascii_digit() {
            prev_underscore = false;
        } else {
            return false;
        }
    }
    true
}

/// `float(s)` for a string: decimal or scientific notation (underscores between digits), `inf`,
/// `nan`.
pub fn py_float(s: &str) -> Option<f64> {
    let t = py_strip(s);
    let (sign, body) = match t.as_bytes().first() {
        Some(b'-') => (-1.0, &t[1..]),
        Some(b'+') => (1.0, &t[1..]),
        _ => (1.0, t),
    };
    let lower = body.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        return Some(sign * f64::INFINITY);
    }
    if lower == "nan" {
        return Some(f64::NAN);
    }
    // mantissa: digits [. digits] | . digits ; exponent: e [sign] digits; underscores only between
    // digits
    let (mant, exp) = match body.find(['e', 'E']) {
        Some(i) => (&body[..i], Some(&body[i + 1..])),
        None => (body, None),
    };
    let (int_part, frac_part) = match mant.find('.') {
        Some(i) => (&mant[..i], Some(&mant[i + 1..])),
        None => (mant, None),
    };
    let ok_int = int_part.is_empty() || digits_with_underscores(int_part);
    let ok_frac = match frac_part {
        None => true,
        Some(f) => f.is_empty() || digits_with_underscores(f),
    };
    let has_digits = !int_part.is_empty() || frac_part.is_some_and(|f| !f.is_empty());
    let ok_exp = match exp {
        None => true,
        Some(e) => {
            let e = e.strip_prefix(['+', '-']).unwrap_or(e);
            digits_with_underscores(e)
        }
    };
    if !(ok_int && ok_frac && has_digits && ok_exp) {
        return None;
    }
    let clean: String = body.chars().filter(|c| *c != '_').collect();
    clean.parse::<f64>().ok().map(|v| sign * v)
}

/// `json.loads(s)` (strict: no control characters inside strings), or `None`; NaN and Infinity are
/// accepted, as by Python, and leave a non-finite value for the callers to reject.
pub fn loads(s: &str) -> Option<PyVal> {
    let mut p = JsonParser {
        s: s.as_bytes(),
        i: 0,
        text: s,
    };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    (p.i == p.s.len()).then_some(v)
}

/// `json.loads(s)` with only finite numbers (`_is_json_finite`), or `None`.
pub fn loads_finite(s: &str) -> Option<PyVal> {
    loads(s).filter(PyVal::is_finite)
}

struct JsonParser<'a> {
    s: &'a [u8],
    i: usize,
    text: &'a str,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Option<PyVal> {
        if depth > 1000 {
            return None;
        }
        match *self.s.get(self.i)? {
            b'{' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat("}") {
                    return Some(PyVal::Dict(items));
                }
                loop {
                    self.ws();
                    if self.s.get(self.i) != Some(&b'"') {
                        return None;
                    }
                    let key = self.string()?;
                    self.ws();
                    if !self.eat(":") {
                        return None;
                    }
                    self.ws();
                    let v = self.value(depth + 1)?;
                    PyVal::dict_insert(&mut items, key, v);
                    self.ws();
                    if self.eat(",") {
                        continue;
                    }
                    if self.eat("}") {
                        return Some(PyVal::Dict(items));
                    }
                    return None;
                }
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat("]") {
                    return Some(PyVal::List(items));
                }
                loop {
                    self.ws();
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    if self.eat(",") {
                        continue;
                    }
                    if self.eat("]") {
                        return Some(PyVal::List(items));
                    }
                    return None;
                }
            }
            b'"' => self.string().map(PyVal::Str),
            b't' if self.eat("true") => Some(PyVal::Bool(true)),
            b'f' if self.eat("false") => Some(PyVal::Bool(false)),
            b'n' if self.eat("null") => Some(PyVal::Null),
            b'N' if self.eat("NaN") => Some(PyVal::Float(f64::NAN)),
            b'I' if self.eat("Infinity") => Some(PyVal::Float(f64::INFINITY)),
            b'-' if self.s[self.i..].starts_with(b"-Infinity") => {
                self.i += 9;
                Some(PyVal::Float(f64::NEG_INFINITY))
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn number(&mut self) -> Option<PyVal> {
        let start = self.i;
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                    self.i += 1;
                }
            }
            _ => return None,
        }
        let mut is_float = false;
        if self.s.get(self.i) == Some(&b'.')
            && self.s.get(self.i + 1).is_some_and(u8::is_ascii_digit)
        {
            is_float = true;
            self.i += 1;
            while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                self.i += 1;
            }
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            let save = self.i;
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                is_float = true;
                while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                    self.i += 1;
                }
            } else {
                self.i = save;
            }
        }
        let text = &self.text[start..self.i];
        if is_float {
            text.parse::<f64>().ok().map(PyVal::Float)
        } else {
            Some(PyVal::Int(normalize_int_digits(text)))
        }
    }

    fn string(&mut self) -> Option<String> {
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            let rest = &self.text[self.i..];
            let mut chars = rest.char_indices();
            let (_, c) = chars.next()?;
            match c {
                '"' => {
                    self.i += 1;
                    return Some(out);
                }
                '\\' => {
                    let e = *self.s.get(self.i + 1)?;
                    self.i += 2;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            if (0xd800..0xdc00).contains(&hi)
                                && self.s[self.i..].starts_with(b"\\u")
                            {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    let cp = 0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                                    out.push(char::from_u32(cp)?);
                                    continue;
                                }
                                self.i = save;
                            }
                            // a lone surrogate has no Rust char: refuse (Python would keep it)
                            out.push(char::from_u32(hi)?);
                        }
                        _ => return None,
                    }
                }
                c if (c as u32) < 0x20 => return None, // strict: control characters must be escaped
                c => {
                    out.push(c);
                    self.i += c.len_utf8();
                }
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = self.text.get(self.i..self.i + 4)?;
        let v = u32::from_str_radix(h, 16).ok()?;
        self.i += 4;
        Some(v)
    }
}

/// `ast.literal_eval(s)` for the containers [F8] accepts, as JSON data (`_jsonable`): `None` for
/// anything else (names, calls, sets, bytes, complex numbers, non-string dict keys, non-finite
/// floats).
pub fn python_literal(text: &str) -> Option<PyVal> {
    let s = py_strip(text);
    if s.chars().count() < 2 || !matches!(s.as_bytes()[0], b'[' | b'{' | b'(') {
        return None;
    }
    let mut p = LitParser { t: s, i: 0 };
    p.ws();
    let v = p.expr(0)?;
    p.ws();
    (p.i == s.len()).then_some(v).filter(PyVal::is_finite)
}

struct LitParser<'a> {
    t: &'a str,
    i: usize,
}

impl LitParser<'_> {
    fn peek(&self) -> Option<char> {
        self.t[self.i..].chars().next()
    }

    fn ws(&mut self) {
        loop {
            let Some(c) = self.peek() else { return };
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\u{c}' {
                self.i += 1;
            } else if c == '\\' && self.t[self.i..].starts_with("\\\n") {
                self.i += 2;
            } else if c == '#' {
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.i += c.len_utf8();
                }
            } else {
                return;
            }
        }
    }

    fn expr(&mut self, depth: usize) -> Option<PyVal> {
        if depth > 64 {
            return None;
        }
        self.ws();
        let c = self.peek()?;
        match c {
            '[' => {
                self.i += 1;
                let items = self.seq(']', depth)?;
                Some(PyVal::List(items))
            }
            '(' => {
                self.i += 1;
                self.ws();
                if self.peek() == Some(')') {
                    self.i += 1;
                    return Some(PyVal::List(Vec::new()));
                }
                let first = self.expr(depth + 1)?;
                self.ws();
                match self.peek()? {
                    ')' => {
                        // a parenthesized expression, not a tuple
                        self.i += 1;
                        Some(first)
                    }
                    ',' => {
                        self.i += 1;
                        let mut items = vec![first];
                        items.extend(self.seq(')', depth)?);
                        Some(PyVal::List(items))
                    }
                    _ => None,
                }
            }
            '{' => {
                self.i += 1;
                self.ws();
                if self.peek() == Some('}') {
                    self.i += 1;
                    return Some(PyVal::Dict(Vec::new()));
                }
                let mut items = Vec::new();
                loop {
                    self.ws();
                    if self.peek() == Some('}') {
                        self.i += 1;
                        return Some(PyVal::Dict(items));
                    }
                    let key = self.expr(depth + 1)?;
                    self.ws();
                    if self.peek() != Some(':') {
                        return None; // a set, or garbage
                    }
                    self.i += 1;
                    let value = self.expr(depth + 1)?;
                    let PyVal::Str(key) = key else { return None };
                    PyVal::dict_insert(&mut items, key, value);
                    self.ws();
                    match self.peek()? {
                        ',' => self.i += 1,
                        '}' => {
                            self.i += 1;
                            return Some(PyVal::Dict(items));
                        }
                        _ => return None,
                    }
                }
            }
            '\'' | '"' => self.strings(),
            'r' | 'R' | 'u' | 'U'
                if matches!(self.t[self.i + 1..].chars().next(), Some('\'' | '"')) =>
            {
                self.strings()
            }
            '-' | '+' => {
                self.i += 1;
                let v = self.expr(depth + 1)?;
                match (c, v) {
                    ('+', v @ (PyVal::Int(_) | PyVal::Float(_))) => Some(v),
                    ('-', PyVal::Int(d)) => Some(PyVal::Int(if d == "0" {
                        d
                    } else if let Some(p) = d.strip_prefix('-') {
                        p.into()
                    } else {
                        format!("-{d}")
                    })),
                    ('-', PyVal::Float(f)) => Some(PyVal::Float(-f)),
                    _ => None,
                }
            }
            '0'..='9' | '.' => self.number(),
            _ => {
                let word: String = self.t[self.i..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                let v = match word.as_str() {
                    "None" => PyVal::Null,
                    "True" => PyVal::Bool(true),
                    "False" => PyVal::Bool(false),
                    _ => return None,
                };
                self.i += word.len();
                Some(v)
            }
        }
    }

    fn seq(&mut self, close: char, depth: usize) -> Option<Vec<PyVal>> {
        let mut items = Vec::new();
        loop {
            self.ws();
            if self.peek() == Some(close) {
                self.i += 1;
                return Some(items);
            }
            items.push(self.expr(depth + 1)?);
            self.ws();
            match self.peek()? {
                ',' => self.i += 1,
                c if c == close => {
                    self.i += 1;
                    return Some(items);
                }
                _ => return None,
            }
        }
    }

    /// One or more adjacent string literals ('a' 'b' is 'ab').
    fn strings(&mut self) -> Option<PyVal> {
        let mut out = String::new();
        loop {
            self.ws();
            let rest = &self.t[self.i..];
            let (raw, skip) = match rest.as_bytes().first() {
                Some(b'r' | b'R') => (true, 1),
                Some(b'u' | b'U') => (false, 1),
                Some(b'\'' | b'"') => (false, 0),
                _ => break,
            };
            if skip == 1 && !matches!(rest.as_bytes().get(1), Some(b'\'' | b'"')) {
                break;
            }
            self.i += skip;
            out.push_str(&self.string(raw)?);
        }
        Some(PyVal::Str(out))
    }

    fn string(&mut self, raw: bool) -> Option<String> {
        let rest = &self.t[self.i..];
        let q = rest.chars().next()?;
        let triple: String = std::iter::repeat_n(q, 3).collect();
        let (delim, mut j) = if rest.starts_with(&triple) {
            (triple.as_str(), 3)
        } else {
            (&rest[..1], 1)
        };
        let long = delim.len() == 3;
        let mut out = String::new();
        loop {
            let tail = &rest[j..];
            if tail.starts_with(delim) {
                self.i += j + delim.len();
                return Some(out);
            }
            let c = tail.chars().next()?;
            if c == '\n' && !long {
                return None;
            }
            if c == '\\' {
                let e = tail[1..].chars().next()?;
                if raw {
                    out.push('\\');
                    out.push(e);
                    j += 1 + e.len_utf8();
                    continue;
                }
                j += 1 + e.len_utf8();
                match e {
                    '\n' => {}
                    '\\' => out.push('\\'),
                    '\'' => out.push('\''),
                    '"' => out.push('"'),
                    'a' => out.push('\u{7}'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'v' => out.push('\u{b}'),
                    'x' => {
                        let h = rest.get(j..j + 2)?;
                        out.push(char::from_u32(u32::from_str_radix(h, 16).ok()?)?);
                        j += 2;
                    }
                    'u' => {
                        let h = rest.get(j..j + 4)?;
                        out.push(char::from_u32(u32::from_str_radix(h, 16).ok()?)?);
                        j += 4;
                    }
                    'U' => {
                        let h = rest.get(j..j + 8)?;
                        out.push(char::from_u32(u32::from_str_radix(h, 16).ok()?)?);
                        j += 8;
                    }
                    '0'..='7' => {
                        let mut v = e.to_digit(8)?;
                        for _ in 0..2 {
                            match rest[j..].chars().next() {
                                Some(d @ '0'..='7') => {
                                    v = v * 8 + d.to_digit(8)?;
                                    j += 1;
                                }
                                _ => break,
                            }
                        }
                        out.push(char::from_u32(v)?);
                    }
                    other => {
                        // an unknown escape keeps its backslash
                        out.push('\\');
                        out.push(other);
                    }
                }
                continue;
            }
            out.push(c);
            j += c.len_utf8();
        }
    }

    fn number(&mut self) -> Option<PyVal> {
        let rest = &self.t[self.i..];
        let len = rest
            .char_indices()
            .take_while(|(k, c)| {
                c.is_ascii_alphanumeric()
                    || *c == '_'
                    || *c == '.'
                    || ((*c == '+' || *c == '-')
                        && *k > 0
                        && matches!(rest.as_bytes()[k - 1], b'e' | b'E'))
            })
            .map(|(k, c)| k + c.len_utf8())
            .last()?;
        let tok = &rest[..len];
        self.i += len;
        let lower = tok.to_ascii_lowercase();
        if lower.ends_with('j') {
            return None; // complex
        }
        for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
            if let Some(body) = lower.strip_prefix(prefix) {
                let body = body.strip_prefix('_').unwrap_or(body);
                if body.is_empty()
                    || body.starts_with('_')
                    || body.ends_with('_')
                    || body.contains("__")
                {
                    return None;
                }
                let digits: String = body.chars().filter(|c| *c != '_').collect();
                let v = u128::from_str_radix(&digits, radix).ok()?;
                return Some(PyVal::Int(v.to_string()));
            }
        }
        if !lower.contains(['.', 'e']) {
            // a decimal integer: no leading zeros but 0 itself (0_0 too)
            let digits: String = tok.chars().filter(|c| *c != '_').collect();
            if digits.len() > 1 && digits.starts_with('0') && digits.chars().any(|c| c != '0') {
                return None;
            }
            return py_int(tok).map(PyVal::Int);
        }
        py_float(tok).map(PyVal::Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        for (f, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (9999999999999998.0, "9999999999999998.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-2.5, "-2.5"),
            (123.456, "123.456"),
            (1e-300, "1e-300"),
            (-0.0, "-0.0"),
            (1e3, "1000.0"),
        ] {
            assert_eq!(float_repr(f), want, "{f}");
        }
    }

    #[test]
    fn dumps_matches_python() {
        let v = PyVal::Dict(vec![
            ("a".into(), PyVal::Str("x\"y\\\n\t\u{1}é".into())),
            (
                "b".into(),
                PyVal::List(vec![PyVal::Int("1".into()), PyVal::Float(2.0), PyVal::Null]),
            ),
        ]);
        assert_eq!(
            dumps(&v),
            "{\"a\": \"x\\\"y\\\\\\n\\t\\u0001é\", \"b\": [1, 2.0, null]}"
        );
    }

    #[test]
    fn numbers_parse_as_python() {
        assert_eq!(py_int(" 7 "), Some("7".into()));
        assert_eq!(py_int("1_000"), Some("1000".into()));
        assert_eq!(py_int("3.0"), None);
        assert_eq!(py_int("-0"), Some("0".into()));
        assert_eq!(py_float("1e3"), Some(1000.0));
        assert_eq!(py_float(".5"), Some(0.5));
        assert_eq!(py_float("5."), Some(5.0));
        assert!(py_float("1__0").is_none());
        assert!(py_float("abc").is_none());
        assert!(py_float("inf").is_some_and(f64::is_infinite));
    }

    #[test]
    fn literal_eval_containers() {
        assert_eq!(
            python_literal("{'x': None, 'y': True}"),
            Some(PyVal::Dict(vec![
                ("x".into(), PyVal::Null),
                ("y".into(), PyVal::Bool(true))
            ]))
        );
        assert_eq!(
            python_literal("['a', 'b']"),
            Some(PyVal::List(vec![
                PyVal::Str("a".into()),
                PyVal::Str("b".into())
            ]))
        );
        assert_eq!(
            python_literal("{\"a\": 1,}"),
            Some(PyVal::Dict(vec![("a".into(), PyVal::Int("1".into()))]))
        );
        assert_eq!(
            python_literal("(1, 2)"),
            Some(PyVal::List(vec![
                PyVal::Int("1".into()),
                PyVal::Int("2".into())
            ]))
        );
        assert_eq!(python_literal("{1, 2}"), None);
        assert_eq!(python_literal("[x]"), None);
    }

    #[test]
    fn loads_is_strict_and_keeps_big_ints() {
        assert_eq!(
            loads("123456789012345678901234567890"),
            Some(PyVal::Int("123456789012345678901234567890".into()))
        );
        assert_eq!(loads("\"a\nb\""), None);
        assert_eq!(
            loads(" [1, 2.5] "),
            Some(PyVal::List(vec![PyVal::Int("1".into()), PyVal::Float(2.5)]))
        );
        assert!(loads_finite("[1e999]").is_none());
    }
}
