// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python's string formatting with runs: `fmt % args` and `fmt.format(...)`. Placeholders' values
//! keep their runs, the format string's literal text keeps its own, and whatever the formatting
//! writes itself (padding, numbers, quotes) is template text.

use std::rc::Rc;

use vllm_parser::python::float_repr;

use super::eval::getattr;
use super::sym::{Kind, Sym, SymBuilder};
use super::value::{VResult, Value, type_error};

fn num_string(v: &Value, spec_type: char, prec: Option<usize>, alt: bool) -> VResult<String> {
    let as_f = || {
        v.as_number()
            .ok_or_else(|| type_error(format!("must be real number, not {}", v.type_name())))
    };
    let as_i = || -> VResult<i64> {
        match v {
            Value::Float(f) => Ok(f.trunc() as i64),
            _ => v.as_int().ok_or_else(|| {
                type_error(format!(
                    "%{spec_type} format: a real number is required, not {}",
                    v.type_name()
                ))
            }),
        }
    };
    Ok(match spec_type {
        'd' | 'i' | 'u' => as_i()?.to_string(),
        'x' => format!("{}{:x}", if alt { "0x" } else { "" }, as_i()?),
        'X' => format!("{}{:X}", if alt { "0X" } else { "" }, as_i()?),
        'o' => format!("{}{:o}", if alt { "0o" } else { "" }, as_i()?),
        'b' => format!("{:b}", as_i()?),
        'f' | 'F' => format!("{:.*}", prec.unwrap_or(6), as_f()?),
        'e' | 'E' => {
            let s = format!("{:.*e}", prec.unwrap_or(6), as_f()?);
            let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
            let exp: i32 = e.parse().unwrap_or(0);
            let r = format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
            if spec_type == 'E' {
                r.to_uppercase()
            } else {
                r
            }
        }
        'g' | 'G' => {
            let f = as_f()?;
            let p = prec.unwrap_or(6).max(1);
            let exp = if f == 0.0 {
                0
            } else {
                f.abs().log10().floor() as i32
            };
            let r = if exp < -4 || exp >= p as i32 {
                let s = format!("{:.*e}", p - 1, f);
                let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
                let m = if m.contains('.') {
                    m.trim_end_matches('0').trim_end_matches('.')
                } else {
                    m
                };
                let e: i32 = e.parse().unwrap_or(0);
                format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
            } else {
                let s = format!("{:.*}", (p as i32 - 1 - exp).max(0) as usize, f);
                if s.contains('.') {
                    s.trim_end_matches('0').trim_end_matches('.').to_string()
                } else {
                    s
                }
            };
            if spec_type == 'G' {
                r.to_uppercase()
            } else {
                r
            }
        }
        '%' => format!("{:.*}%", prec.unwrap_or(6), as_f()? * 100.0),
        'c' => match v {
            Value::Int(i) => char::from_u32(*i as u32).map(String::from).unwrap_or_default(),
            Value::Str(s) => s.as_str().to_string(),
            _ => return Err(type_error("%c requires int or char")),
        },
        _ => {
            return Err(type_error(format!(
                "unsupported format character '{spec_type}'"
            )));
        }
    })
}

/// Pad `body` to `width` (left-aligned when `left`), with `fill` (template text).
fn pad(body: Sym, width: usize, left: bool, fill: char) -> Sym {
    let n = body.len_chars();
    if n >= width {
        return body;
    }
    let f = Sym::template(&fill.to_string());
    if left {
        body.pad(0, width - n, &f)
    } else {
        body.pad(width - n, 0, &f)
    }
}

/// Signs, zero padding and width for a number's digits.
fn pad_number(
    digits: String,
    plus: bool,
    space: bool,
    zero: bool,
    left: bool,
    width: usize,
) -> Sym {
    let (neg, abs) = match digits.strip_prefix('-') {
        Some(rest) => (true, rest.to_string()),
        None => (false, digits),
    };
    let sign = if neg {
        "-"
    } else if plus {
        "+"
    } else if space {
        " "
    } else {
        ""
    };
    let body_len = sign.len() + abs.chars().count();
    let text = if zero && !left && body_len < width {
        format!("{sign}{}{abs}", "0".repeat(width - body_len))
    } else {
        format!("{sign}{abs}")
    };
    pad(Sym::template(&text), width, left, ' ')
}

/// `fmt % args`.
pub(crate) fn percent(fmt: &Sym, args: &Value) -> VResult<Sym> {
    let v = fmt.as_str();
    let mapping = matches!(args, Value::Dict(_));
    let positional: Vec<Value> = match args {
        Value::Tuple(items) => items.as_ref().clone(),
        Value::Dict(_) => Vec::new(),
        other => vec![other.clone()],
    };
    let mut next = 0;
    let mut take = || -> VResult<Value> {
        let v = positional
            .get(next)
            .cloned()
            .ok_or_else(|| type_error("not enough arguments for format string"))?;
        next += 1;
        Ok(v)
    };
    let mut b = SymBuilder::new();
    let bytes = v.as_bytes();
    let mut pos = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            i += 1;
            continue;
        }
        b.push_slice(fmt, pos, i);
        let mut j = i + 1;
        let mut key = None;
        if bytes.get(j) == Some(&b'(') {
            let close = v[j..].find(')').ok_or_else(|| type_error("incomplete format key"))? + j;
            key = Some(v[j + 1..close].to_string());
            j = close + 1;
        }
        let mut flags = String::new();
        while let Some(&c) = bytes.get(j) {
            if b"-+ #0".contains(&c) {
                flags.push(c as char);
                j += 1;
            } else {
                break;
            }
        }
        let mut width = 0usize;
        if bytes.get(j) == Some(&b'*') {
            width = take()?.as_int().unwrap_or(0).max(0) as usize;
            j += 1;
        } else {
            while let Some(c) = bytes.get(j).filter(|c| c.is_ascii_digit()) {
                width = width * 10 + usize::from(c - b'0');
                j += 1;
            }
        }
        let mut prec = None;
        if bytes.get(j) == Some(&b'.') {
            j += 1;
            if bytes.get(j) == Some(&b'*') {
                prec = Some(take()?.as_int().unwrap_or(0).max(0) as usize);
                j += 1;
            } else {
                let mut p = 0usize;
                while let Some(c) = bytes.get(j).filter(|c| c.is_ascii_digit()) {
                    p = p * 10 + usize::from(c - b'0');
                    j += 1;
                }
                prec = Some(p);
            }
        }
        while bytes.get(j).is_some_and(|c| b"hlL".contains(c)) {
            j += 1;
        }
        let Some(&t) = bytes.get(j) else {
            return Err(type_error("incomplete format"));
        };
        let t = t as char;
        j += 1;
        if t == '%' {
            b.push(Kind::Template, "%");
            pos = j;
            i = j;
            continue;
        }
        let value = match (&key, args) {
            (Some(k), Value::Dict(d)) => {
                d.get_str(k).cloned().ok_or_else(|| type_error(format!("'{k}'")))?
            }
            (Some(_), _) => return Err(type_error("format requires a mapping")),
            (None, _) if mapping => args.clone(),
            (None, _) => take()?,
        };
        let left = flags.contains('-');
        let piece = match t {
            's' | 'r' | 'a' => {
                let mut body = if t == 's' {
                    value.to_sym()
                } else {
                    value.repr()
                };
                if let Some(p) = prec {
                    body = body.slice_chars(0, p.min(body.len_chars()));
                }
                pad(body, width, left, ' ')
            }
            _ => {
                let digits = num_string(&value, t, prec, flags.contains('#'))?;
                pad_number(
                    digits,
                    flags.contains('+'),
                    flags.contains(' '),
                    flags.contains('0'),
                    left,
                    width,
                )
            }
        };
        b.push_sym(&piece);
        pos = j;
        i = j;
    }
    if !mapping && next < positional.len() && !matches!(args, Value::Dict(_)) {
        return Err(type_error(
            "not all arguments converted during string formatting",
        ));
    }
    b.push_slice(fmt, pos, v.len());
    Ok(b.finish())
}

/// A format spec: `[[fill]align][sign][#][0][width][,_][.precision][type]`.
struct Spec {
    fill: char,
    align: Option<char>,
    sign: Option<char>,
    alt: bool,
    zero: bool,
    width: usize,
    grouping: Option<char>,
    precision: Option<usize>,
    ty: Option<char>,
}

fn parse_spec(spec: &str) -> VResult<Spec> {
    let chars: Vec<char> = spec.chars().collect();
    let mut i = 0;
    let mut s = Spec {
        fill: ' ',
        align: None,
        sign: None,
        alt: false,
        zero: false,
        width: 0,
        grouping: None,
        precision: None,
        ty: None,
    };
    if chars.len() >= 2 && "<>=^".contains(chars[1]) {
        s.fill = chars[0];
        s.align = Some(chars[1]);
        i = 2;
    } else if !chars.is_empty() && "<>=^".contains(chars[0]) {
        s.align = Some(chars[0]);
        i = 1;
    }
    if i < chars.len() && "+- ".contains(chars[i]) {
        s.sign = Some(chars[i]);
        i += 1;
    }
    if i < chars.len() && chars[i] == '#' {
        s.alt = true;
        i += 1;
    }
    if i < chars.len() && chars[i] == '0' {
        s.zero = true;
        i += 1;
    }
    while i < chars.len() && chars[i].is_ascii_digit() {
        s.width = s.width * 10 + chars[i].to_digit(10).unwrap_or(0) as usize;
        i += 1;
    }
    if i < chars.len() && ",_".contains(chars[i]) {
        s.grouping = Some(chars[i]);
        i += 1;
    }
    if i < chars.len() && chars[i] == '.' {
        i += 1;
        let mut p = 0;
        while i < chars.len() && chars[i].is_ascii_digit() {
            p = p * 10 + chars[i].to_digit(10).unwrap_or(0) as usize;
            i += 1;
        }
        s.precision = Some(p);
    }
    if i < chars.len() {
        s.ty = Some(chars[i]);
        i += 1;
    }
    if i != chars.len() {
        return Err(type_error("Invalid format specifier"));
    }
    Ok(s)
}

/// `format(value, spec)` with runs.
pub(crate) fn format_value(value: &Value, spec: &str) -> VResult<Sym> {
    if spec.is_empty() {
        return Ok(value.to_sym());
    }
    let s = parse_spec(spec)?;
    match value {
        Value::Str(text) => {
            if s.ty.is_some_and(|t| t != 's') {
                return Err(type_error(format!(
                    "Unknown format code '{}' for object of type 'str'",
                    s.ty.unwrap_or('?')
                )));
            }
            let body = match s.precision {
                Some(p) => text.slice_chars(0, p.min(text.len_chars())),
                None => text.clone(),
            };
            let n = body.len_chars();
            let marg = s.width.saturating_sub(n);
            let (l, r) = match s.align.unwrap_or('<') {
                '>' => (marg, 0),
                '^' => (marg / 2, marg - marg / 2),
                _ => (0, marg),
            };
            Ok(body.pad(l, r, &Sym::template(&s.fill.to_string())))
        }
        v if v.as_number().is_some() => {
            let ty = s.ty.unwrap_or(if matches!(v, Value::Float(_)) {
                'r'
            } else {
                'd'
            });
            let mut digits = match ty {
                'r' => float_repr(v.as_number().unwrap_or(0.0)),
                'n' => num_string(v, 'd', None, false)?,
                other => num_string(v, other, s.precision, s.alt)?,
            };
            if let Some(g) = s.grouping {
                let (neg, rest) =
                    digits.strip_prefix('-').map_or((false, digits.as_str()), |r| (true, r));
                let (int_part, frac) =
                    rest.split_once('.').map_or((rest, None), |(a, b)| (a, Some(b)));
                let mut grouped = String::new();
                for (k, c) in int_part.chars().enumerate() {
                    if k > 0 && (int_part.len() - k) % 3 == 0 {
                        grouped.push(g);
                    }
                    grouped.push(c);
                }
                digits = format!(
                    "{}{grouped}{}",
                    if neg { "-" } else { "" },
                    frac.map(|f| format!(".{f}")).unwrap_or_default()
                );
            }
            let left = s.align == Some('<');
            Ok(pad_number(
                digits,
                s.sign == Some('+'),
                s.sign == Some(' '),
                s.zero || s.align == Some('='),
                left,
                s.width,
            ))
        }
        other => Ok(other.to_sym()),
    }
}

/// Resolve a field name (`0`, `name`, `name.attr`, `name[key]`).
fn field_value(
    name: &str,
    pos: &[Value],
    kw: &[(Rc<str>, Value)],
    auto: &mut usize,
) -> VResult<Value> {
    let end = name.find(['.', '[']).unwrap_or(name.len());
    let (first, mut rest) = name.split_at(end);
    let mut v = if first.is_empty() {
        let v = pos
            .get(*auto)
            .cloned()
            .ok_or_else(|| type_error("Replacement index out of range"))?;
        *auto += 1;
        v
    } else if let Ok(i) = first.parse::<usize>() {
        pos.get(i)
            .cloned()
            .ok_or_else(|| type_error(format!("Replacement index {i} out of range")))?
    } else {
        kw.iter()
            .find(|(k, _)| &**k == first)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| type_error(format!("'{first}'")))?
    };
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix('.') {
            let e = r.find(['.', '[']).unwrap_or(r.len());
            v = getattr(&v, &r[..e])?;
            rest = &r[e..];
        } else if let Some(r) = rest.strip_prefix('[') {
            let e = r.find(']').ok_or_else(|| type_error("Missing ']' in format string"))?;
            let key = &r[..e];
            let k = key.parse::<i64>().map(Value::Int).unwrap_or_else(|_| Value::template_str(key));
            v = super::eval::getitem(&v, &k)?;
            rest = &r[e + 1..];
        } else {
            return Err(type_error(
                "Only '.' or '[' may follow ']' in format field specifier",
            ));
        }
    }
    Ok(v)
}

/// `fmt.format(*pos, **kw)`.
pub(crate) fn str_format(fmt: &Sym, pos: &[Value], kw: &[(Rc<str>, Value)]) -> VResult<Sym> {
    let v = fmt.as_str();
    let bytes = v.as_bytes();
    let mut b = SymBuilder::new();
    let mut auto = 0;
    let mut i = 0;
    let mut lit_start = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c != b'{' && c != b'}' {
            i += 1;
            continue;
        }
        if bytes.get(i + 1) == Some(&c) {
            b.push_slice(fmt, lit_start, i + 1);
            i += 2;
            lit_start = i;
            continue;
        }
        if c == b'}' {
            return Err(type_error("Single '}' encountered in format string"));
        }
        b.push_slice(fmt, lit_start, i);
        let mut depth = 1;
        let mut j = i + 1;
        let mut bracket = false;
        while j < bytes.len() && depth > 0 {
            match bytes[j] {
                b'[' => bracket = true,
                b']' => bracket = false,
                b'{' if !bracket => depth += 1,
                b'}' if !bracket => depth -= 1,
                _ => {}
            }
            j += 1;
        }
        if depth > 0 {
            return Err(type_error("expected '}' before end of string"));
        }
        let body = &v[i + 1..j - 1];
        let mut name_end = body.len();
        let mut br = false;
        for (q, ch) in body.char_indices() {
            match ch {
                '[' => br = true,
                ']' => br = false,
                '!' | ':' if !br => {
                    name_end = q;
                    break;
                }
                _ => {}
            }
        }
        let (name, rest) = body.split_at(name_end);
        let (conv, spec) = match rest.strip_prefix('!') {
            Some(r) => (
                r.chars().next(),
                r.get(1..).unwrap_or("").strip_prefix(':').unwrap_or(""),
            ),
            None => (None, rest.strip_prefix(':').unwrap_or("")),
        };
        let mut value = field_value(name, pos, kw, &mut auto)?;
        match conv {
            Some('r') | Some('a') => value = Value::Str(value.repr()),
            Some('s') => value = Value::Str(value.to_sym()),
            _ => {}
        }
        let spec = if spec.contains('{') {
            str_format(&Sym::template(spec), pos, kw)?.as_str().to_string()
        } else {
            spec.to_string()
        };
        b.push_sym(&format_value(&value, &spec)?);
        i = j;
        lit_start = j;
    }
    b.push_slice(fmt, lit_start, v.len());
    Ok(b.finish())
}
