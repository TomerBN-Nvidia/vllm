// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `Sym`: a string of runs, each remembering where its characters came from: template text, client
//! data, text of unknown origin, or a control token the template wrote (the Python plugin's `Sym`,
//! a `str` subclass there). Its value is what the model reads (control tokens spelled out), and
//! every string operation gives Python `str`'s result run by run, with one rule: client text never
//! equals a control token (a match that lines one up with the other is no match). Text of unknown
//! origin that meets a control token is an error ([`SymError::Unknown`]): the encoder can't tell.
//!
//! Indices are Python's: code points, not bytes.

use std::fmt;
use std::rc::Rc;

use vllm_parser::python::py_isspace;

/// Where a run's characters came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Kind {
    /// The template's own text (and the chat schema's fields: keys, roles, types).
    Template,
    /// Client data: never a control token.
    Data,
    /// Text of unknown origin: an operation the encoder doesn't model made it.
    Unknown,
    /// A control token the template wrote: its id; the run's text is its spelling.
    Control(u32),
}

impl Kind {
    fn bit(self) -> u8 {
        match self {
            Kind::Template => 1,
            Kind::Data => 2,
            Kind::Unknown => 4,
            Kind::Control(_) => 8,
        }
    }

    /// The kind of a part of this run's text: a control token cut out of its spelling is template
    /// text.
    fn cut(self) -> Kind {
        match self {
            Kind::Control(_) => Kind::Template,
            other => other,
        }
    }
}

const BIT_C: u8 = 8;
const BIT_DU: u8 = 2 | 4;

/// An error a string operation raises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SymError {
    /// Text of unknown origin lines up with a control token.
    Unknown(String),
    /// Python's own error (`ValueError`, `TypeError`, ...), as its message.
    Python(String),
}

impl fmt::Display for SymError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymError::Unknown(message) | SymError::Python(message) => f.write_str(message),
        }
    }
}

pub(crate) type SymResult<T> = Result<T, SymError>;

/// One run: its end (a byte offset into the text) and kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Run {
    end: usize,
    kind: Kind,
}

/// A string of runs; cheap to clone.
#[derive(Clone)]
pub(crate) struct Sym {
    text: Rc<str>,
    runs: Rc<[Run]>,
    kinds: u8,
}

impl fmt::Debug for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        let mut start = 0;
        for r in self.runs.iter() {
            list.entry(&(r.kind, &self.text[start..r.end]));
            start = r.end;
        }
        list.finish()
    }
}

impl PartialEq for Sym {
    /// Structural identity (same text, same runs): for tests and caches, not Python's `==`.
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text && self.runs == other.runs
    }
}

/// Builds a `Sym` run by run.
#[derive(Default, Debug)]
pub(crate) struct SymBuilder {
    text: String,
    runs: Vec<Run>,
    kinds: u8,
}

impl SymBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append text of `kind` (a control kind keeps its id only when `text` is its whole spelling;
    /// callers pass `Kind::Control` only for whole spellings).
    pub(crate) fn push(&mut self, kind: Kind, text: &str) {
        if text.is_empty() {
            return;
        }
        let merge = !matches!(kind, Kind::Control(_))
            && self.runs.last().is_some_and(|last| last.kind == kind);
        self.text.push_str(text);
        if merge {
            self.runs.last_mut().expect("checked").end = self.text.len();
        } else {
            self.runs.push(Run {
                end: self.text.len(),
                kind,
            });
        }
        self.kinds |= kind.bit();
    }

    /// Append all of `s`.
    pub(crate) fn push_sym(&mut self, s: &Sym) {
        for (kind, text) in s.runs() {
            self.push(kind, text);
        }
    }

    /// Append `s[a..b]` (byte offsets on char boundaries); a control token cut by the bounds
    /// becomes template text.
    pub(crate) fn push_slice(&mut self, s: &Sym, a: usize, b: usize) {
        if a >= b {
            return;
        }
        let mut start = 0;
        for r in s.runs.iter() {
            let (lo, hi) = (start.max(a), r.end.min(b));
            if lo < hi {
                let kind = if lo == start && hi == r.end {
                    r.kind
                } else {
                    r.kind.cut()
                };
                self.push(kind, &s.text[lo..hi]);
            }
            if r.end >= b {
                break;
            }
            start = r.end;
        }
    }

    pub(crate) fn finish(self) -> Sym {
        Sym {
            text: Rc::from(self.text),
            runs: Rc::from(self.runs),
            kinds: self.kinds,
        }
    }
}

/// How a match lines control tokens up with client text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Clash {
    /// Not at all: a match.
    None,
    /// Client data meets a control token: no match.
    Data,
    /// Text of unknown origin meets one: the encoder can't tell.
    Unknown,
}

fn byte_of_char(text: &str, chars: usize) -> usize {
    if text.is_ascii() {
        return chars.min(text.len());
    }
    text.char_indices().nth(chars).map_or(text.len(), |(b, _)| b)
}

fn char_of_byte(text: &str, byte: usize) -> usize {
    if text.is_ascii() {
        byte
    } else {
        text[..byte].chars().count()
    }
}

/// Python's line boundaries (`str.splitlines`).
fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\x0b'
            | '\x0c'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

impl Sym {
    pub(crate) fn empty() -> Sym {
        SymBuilder::new().finish()
    }

    /// One run of `kind`.
    pub(crate) fn of(kind: Kind, text: &str) -> Sym {
        let mut b = SymBuilder::new();
        b.push(kind, text);
        b.finish()
    }

    pub(crate) fn template(text: &str) -> Sym {
        Sym::of(Kind::Template, text)
    }

    pub(crate) fn data(text: &str) -> Sym {
        Sym::of(Kind::Data, text)
    }

    pub(crate) fn unknown(text: &str) -> Sym {
        Sym::of(Kind::Unknown, text)
    }

    /// What the model reads.
    pub(crate) fn as_str(&self) -> &str {
        &self.text
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Python's `len`: code points.
    pub(crate) fn len_chars(&self) -> usize {
        if self.text.is_ascii() {
            self.text.len()
        } else {
            self.text.chars().count()
        }
    }

    /// (kind, text) of each run.
    pub(crate) fn runs(&self) -> impl Iterator<Item = (Kind, &str)> + '_ {
        let mut start = 0;
        self.runs.iter().map(move |r| {
            let text = &self.text[start..r.end];
            start = r.end;
            (r.kind, text)
        })
    }

    /// `self[a..b]` by byte offsets.
    fn slice_bytes(&self, a: usize, b: usize) -> Sym {
        if a == 0 && b >= self.text.len() {
            return self.clone();
        }
        let mut builder = SymBuilder::new();
        builder.push_slice(self, a, b);
        builder.finish()
    }

    /// `self[a:b]` by Python (code point) indices, already clamped: `0 <= a <= b <= len`.
    pub(crate) fn slice_chars(&self, a: usize, b: usize) -> Sym {
        let (ba, bb) = (byte_of_char(&self.text, a), byte_of_char(&self.text, b));
        self.slice_bytes(ba, bb.max(ba))
    }

    /// `self[start:stop:step]` with Python's slice semantics (`None` bounds as Python's).
    pub(crate) fn slice(
        &self,
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> SymResult<Sym> {
        let n = self.len_chars() as i64;
        let step = step.unwrap_or(1);
        if step == 0 {
            return Err(SymError::Python("slice step cannot be zero".into()));
        }
        let (start, stop) = slice_bounds(n, start, stop, step);
        if step == 1 {
            return Ok(self.slice_chars(start as usize, stop.max(start) as usize));
        }
        let chars = self.char_kinds();
        let mut b = SymBuilder::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            let (c, k) = chars[i as usize];
            let mut buf = [0u8; 4];
            b.push(k.cut(), c.encode_utf8(&mut buf));
            i += step;
        }
        Ok(b.finish())
    }

    /// `self[i]` (a Python index).
    pub(crate) fn index_char(&self, i: i64) -> SymResult<Sym> {
        let n = self.len_chars() as i64;
        let i = if i < 0 { i + n } else { i };
        if !(0..n).contains(&i) {
            return Err(SymError::Python("string index out of range".into()));
        }
        Ok(self.slice_chars(i as usize, i as usize + 1))
    }

    /// Each character with its run's kind.
    fn char_kinds(&self) -> Vec<(char, Kind)> {
        let mut out = Vec::with_capacity(self.text.len());
        for (kind, text) in self.runs() {
            out.extend(text.chars().map(|c| (c, kind)));
        }
        out
    }

    /// The characters, each a `Sym` (iteration over a string).
    pub(crate) fn chars(&self) -> Vec<Sym> {
        let n = self.len_chars();
        (0..n).map(|i| self.slice_chars(i, i + 1)).collect()
    }

    pub(crate) fn concat(&self, other: &Sym) -> Sym {
        if other.is_empty() {
            return self.clone();
        }
        if self.is_empty() {
            return other.clone();
        }
        let mut b = SymBuilder::new();
        b.push_sym(self);
        b.push_sym(other);
        b.finish()
    }

    pub(crate) fn repeat(&self, n: i64) -> Sym {
        let mut b = SymBuilder::new();
        for _ in 0..n.max(0) {
            b.push_sym(self);
        }
        b.finish()
    }

    /// The run containing byte `p` (its index and start).
    fn run_at(&self, p: usize) -> (usize, usize) {
        let i = self.runs.partition_point(|r| r.end <= p);
        let start = if i == 0 { 0 } else { self.runs[i - 1].end };
        (i, start)
    }

    /// (start, end, kind) of the runs overlapping bytes [a, b), clipped.
    fn segments(&self, a: usize, b: usize) -> Vec<(usize, usize, Kind)> {
        let mut out = Vec::new();
        if a >= b {
            return out;
        }
        let (mut i, mut start) = self.run_at(a);
        while i < self.runs.len() && start < b {
            let r = self.runs[i];
            out.push((start.max(a), r.end.min(b), r.kind));
            start = r.end;
            i += 1;
        }
        out
    }

    /// How a match of `needle` at bytes [p, p + needle.len) lines up control tokens with client
    /// text.
    fn clash(&self, p: usize, needle: &Sym) -> Clash {
        let n = needle.text.len();
        if !((self.kinds & BIT_C != 0 && needle.kinds & BIT_DU != 0)
            || (self.kinds & BIT_DU != 0 && needle.kinds & BIT_C != 0))
        {
            return Clash::None;
        }
        let hs = self.segments(p, p + n);
        let ns: Vec<(usize, usize, Kind)> =
            needle.segments(0, n).into_iter().map(|(a, b, k)| (a + p, b + p, k)).collect();
        let (mut i, mut j) = (0, 0);
        let mut unknown = false;
        while i < hs.len() && j < ns.len() {
            let (a1, b1, k1) = hs[i];
            let (a2, b2, k2) = ns[j];
            let c1 = matches!(k1, Kind::Control(_));
            let c2 = matches!(k2, Kind::Control(_));
            if a1 < b2 && a2 < b1 && (c1 != c2) {
                let other = if c1 { k2 } else { k1 };
                match other {
                    Kind::Data => return Clash::Data,
                    Kind::Unknown => unknown = true,
                    _ => {}
                }
            }
            if b1 <= b2 {
                i += 1;
            } else {
                j += 1;
            }
        }
        if unknown { Clash::Unknown } else { Clash::None }
    }

    fn unknown_error(&self, what: &str, p: usize, n: usize) -> SymError {
        let lo = self.text[..p].char_indices().rev().nth(19).map_or(0, |(b, _)| b);
        let hi = (p + n..=self.text.len())
            .filter(|b| self.text.is_char_boundary(*b))
            .nth(20)
            .unwrap_or(self.text.len());
        SymError::Unknown(format!(
            "nemotron4 structural rendering: {what} lines up text of unknown origin with a control \
             token at {:?}; the chat template uses an operation the encoder doesn't model",
            &self.text[lo..hi]
        ))
    }

    /// Byte positions of `needle` in bytes [a, e) from the left, non-overlapping (or every one,
    /// stepping one character, when `overlapping`), less the ones that line client text up with a
    /// control token.
    fn matches(
        &self,
        needle: &Sym,
        a: usize,
        e: usize,
        what: &str,
        limit: usize,
    ) -> SymResult<Vec<usize>> {
        let mut out = Vec::new();
        let nv = needle.as_str();
        let n = nv.len();
        if n == 0 || e < a {
            return Ok(out);
        }
        let hay = &self.text[..e];
        let mut i = a;
        while out.len() < limit {
            let Some(found) = hay[i..].find(nv) else {
                break;
            };
            let p = i + found;
            match self.clash(p, needle) {
                Clash::None => {
                    out.push(p);
                    i = p + n;
                }
                Clash::Data => i = p + next_char_len(&self.text, p),
                Clash::Unknown => return Err(self.unknown_error(what, p, n)),
            }
            if i > e {
                break;
            }
        }
        Ok(out)
    }

    /// The last match of `needle` in bytes [a, e), as `str.rfind` would find it, less clashing
    /// ones.
    fn last_match(&self, needle: &Sym, a: usize, e: usize, what: &str) -> SymResult<Option<usize>> {
        let nv = needle.as_str();
        let n = nv.len();
        if e < a || e - a < n {
            return Ok(None);
        }
        let mut end = e;
        loop {
            let Some(found) = self.text[a..end].rfind(nv) else {
                return Ok(None);
            };
            let p = a + found;
            match self.clash(p, needle) {
                Clash::None => return Ok(Some(p)),
                Clash::Data => {
                    // the next candidate ends before this one's last character
                    end = p + n - prev_char_len(&self.text, p + n);
                    if end < a + n {
                        return Ok(None);
                    }
                }
                Clash::Unknown => return Err(self.unknown_error(what, p, n)),
            }
        }
    }

    /// Byte bounds of Python's `[start:end]` for search methods.
    fn search_bounds(&self, start: Option<i64>, end: Option<i64>) -> (usize, usize) {
        let n = self.len_chars() as i64;
        let fix = |v: i64| if v < 0 { (v + n).max(0) } else { v.min(n) };
        let s = fix(start.unwrap_or(0)) as usize;
        let e = fix(end.unwrap_or(n)) as usize;
        (byte_of_char(&self.text, s), byte_of_char(&self.text, e))
    }

    /// Python's `==` against another string.
    pub(crate) fn py_eq(&self, other: &Sym) -> SymResult<bool> {
        if self.text != other.text {
            return Ok(false);
        }
        if self.text.is_empty() {
            return Ok(true);
        }
        match self.clash(0, other) {
            Clash::None => Ok(true),
            Clash::Data => Ok(false),
            Clash::Unknown => Err(self.unknown_error("==", 0, self.text.len())),
        }
    }

    /// `needle in self`.
    pub(crate) fn contains(&self, needle: &Sym) -> SymResult<bool> {
        if needle.is_empty() {
            return Ok(true);
        }
        Ok(!self.matches(needle, 0, self.text.len(), "in", 1)?.is_empty())
    }

    pub(crate) fn find(
        &self,
        needle: &Sym,
        start: Option<i64>,
        end: Option<i64>,
    ) -> SymResult<i64> {
        let (a, e) = self.search_bounds(start, end);
        if needle.is_empty() {
            let n = self.len_chars() as i64;
            let s = start.map_or(0, |v| if v < 0 { (v + n).max(0) } else { v });
            return Ok(if s > n { -1 } else { s.min(n) });
        }
        Ok(match self.matches(needle, a, e, "find", 1)?.first() {
            Some(&p) => char_of_byte(&self.text, p) as i64,
            None => -1,
        })
    }

    pub(crate) fn rfind(
        &self,
        needle: &Sym,
        start: Option<i64>,
        end: Option<i64>,
    ) -> SymResult<i64> {
        let (a, e) = self.search_bounds(start, end);
        if needle.is_empty() {
            return Ok(char_of_byte(&self.text, e) as i64);
        }
        Ok(match self.last_match(needle, a, e, "rfind")? {
            Some(p) => char_of_byte(&self.text, p) as i64,
            None => -1,
        })
    }

    pub(crate) fn count(
        &self,
        needle: &Sym,
        start: Option<i64>,
        end: Option<i64>,
    ) -> SymResult<i64> {
        let (a, e) = self.search_bounds(start, end);
        if needle.is_empty() {
            let chars = char_of_byte(&self.text, e) - char_of_byte(&self.text, a);
            return Ok(chars as i64 + 1);
        }
        Ok(self.matches(needle, a, e, "count", usize::MAX)?.len() as i64)
    }

    /// `startswith` / `endswith` with one affix.
    pub(crate) fn affix(
        &self,
        affix: &Sym,
        start: Option<i64>,
        end: Option<i64>,
        at_end: bool,
    ) -> SymResult<bool> {
        let (a, e) = self.search_bounds(start, end);
        let nv = affix.as_str();
        if e < a {
            return Ok(false);
        }
        let window = &self.text[a..e];
        let hit = if at_end {
            window.ends_with(nv)
        } else {
            window.starts_with(nv)
        };
        if !hit {
            return Ok(false);
        }
        if nv.is_empty() {
            return Ok(true);
        }
        let p = if at_end { e - nv.len() } else { a };
        match self.clash(p, affix) {
            Clash::None => Ok(true),
            Clash::Data => Ok(false),
            Clash::Unknown => {
                Err(self.unknown_error(if at_end { "endswith" } else { "startswith" }, p, nv.len()))
            }
        }
    }

    pub(crate) fn split(&self, sep: Option<&Sym>, maxsplit: i64) -> SymResult<Vec<Sym>> {
        let text: &str = &self.text;
        let Some(sep) = sep else {
            // runs of whitespace separate; leading and trailing whitespace give no empty pieces
            let mut bounds = Vec::new();
            let mut i = 0;
            while let Some(off) = text[i..].find(|c: char| !py_isspace(c)) {
                i += off;
                if maxsplit >= 0 && bounds.len() as i64 >= maxsplit {
                    bounds.push((i, text.len()));
                    break;
                }
                let j = text[i..].find(py_isspace).map_or(text.len(), |o| i + o);
                bounds.push((i, j));
                i = j;
            }
            return Ok(bounds.into_iter().map(|(a, b)| self.slice_bytes(a, b)).collect());
        };
        if sep.is_empty() {
            return Err(SymError::Python("empty separator".into()));
        }
        let limit = if maxsplit < 0 {
            usize::MAX
        } else {
            maxsplit as usize
        };
        let hits = self.matches(sep, 0, text.len(), "split", limit)?;
        let mut out = Vec::with_capacity(hits.len() + 1);
        let mut a = 0;
        for p in hits {
            out.push(self.slice_bytes(a, p));
            a = p + sep.text.len();
        }
        out.push(self.slice_bytes(a, text.len()));
        Ok(out)
    }

    pub(crate) fn rsplit(&self, sep: Option<&Sym>, maxsplit: i64) -> SymResult<Vec<Sym>> {
        let text: &str = &self.text;
        let Some(sep) = sep else {
            let mut words = Vec::new();
            let mut i = 0;
            while let Some(off) = text[i..].find(|c: char| !py_isspace(c)) {
                let a = i + off;
                let b = text[a..].find(py_isspace).map_or(text.len(), |o| a + o);
                words.push((a, b));
                i = b;
            }
            if maxsplit >= 0 && (maxsplit as usize) < words.len() {
                let keep = words.len() - maxsplit as usize;
                let first = (0, words[keep - 1].1);
                let mut rest = vec![first];
                rest.extend_from_slice(&words[keep..]);
                // the first piece keeps its leading whitespace stripped, as str's
                let lead = text.find(|c: char| !py_isspace(c)).unwrap_or(0);
                rest[0].0 = lead;
                words = rest;
            }
            return Ok(words.into_iter().map(|(a, b)| self.slice_bytes(a, b)).collect());
        };
        if sep.is_empty() {
            return Err(SymError::Python("empty separator".into()));
        }
        let mut bounds = Vec::new();
        let mut b = text.len();
        let mut end = text.len();
        while maxsplit < 0 || (bounds.len() as i64) < maxsplit {
            let Some(p) = self.last_match(sep, 0, end, "rsplit")? else {
                break;
            };
            bounds.push((p + sep.text.len(), b));
            b = p;
            end = p;
        }
        bounds.push((0, b));
        bounds.reverse();
        Ok(bounds.into_iter().map(|(a, b)| self.slice_bytes(a, b)).collect())
    }

    pub(crate) fn splitlines(&self, keepends: bool) -> Vec<Sym> {
        let text: &str = &self.text;
        let mut out = Vec::new();
        let mut a = 0;
        let mut it = text.char_indices().peekable();
        while let Some((i, c)) = it.next() {
            if !is_line_break(c) {
                continue;
            }
            let mut end = i + c.len_utf8();
            if c == '\r' && it.peek().is_some_and(|(_, n)| *n == '\n') {
                it.next();
                end += 1;
            }
            out.push(self.slice_bytes(a, if keepends { end } else { i }));
            a = end;
        }
        if a < text.len() {
            out.push(self.slice_bytes(a, text.len()));
        }
        out
    }

    pub(crate) fn partition(&self, sep: &Sym, from_right: bool) -> SymResult<(Sym, Sym, Sym)> {
        if sep.is_empty() {
            return Err(SymError::Python("empty separator".into()));
        }
        let n = self.text.len();
        let hit = if from_right {
            self.last_match(sep, 0, n, "rpartition")?
        } else {
            self.matches(sep, 0, n, "partition", 1)?.first().copied()
        };
        Ok(match hit {
            Some(p) => (
                self.slice_bytes(0, p),
                self.slice_bytes(p, p + sep.text.len()),
                self.slice_bytes(p + sep.text.len(), n),
            ),
            None if from_right => (Sym::empty(), Sym::empty(), self.clone()),
            None => (self.clone(), Sym::empty(), Sym::empty()),
        })
    }

    /// `self.join(items)`.
    pub(crate) fn join(&self, items: &[Sym]) -> Sym {
        let mut b = SymBuilder::new();
        for (k, item) in items.iter().enumerate() {
            if k > 0 {
                b.push_sym(self);
            }
            b.push_sym(item);
        }
        b.finish()
    }

    pub(crate) fn replace(&self, old: &Sym, new: &Sym, count: i64) -> SymResult<Sym> {
        let n = self.text.len();
        let limit = if count < 0 {
            usize::MAX
        } else {
            count as usize
        };
        let mut b = SymBuilder::new();
        if old.is_empty() {
            // `new` before every character and at the end
            let mut a = 0;
            for (k, (i, _)) in self.text.char_indices().chain(std::iter::once((n, ' '))).enumerate()
            {
                if k >= limit {
                    break;
                }
                b.push_slice(self, a, i);
                b.push_sym(new);
                a = i;
            }
            b.push_slice(self, a, n);
            return Ok(b.finish());
        }
        let mut a = 0;
        for p in self.matches(old, 0, n, "replace", limit)? {
            b.push_slice(self, a, p);
            b.push_sym(new);
            a = p + old.text.len();
        }
        b.push_slice(self, a, n);
        Ok(b.finish())
    }

    /// `strip` / `lstrip` / `rstrip`.
    pub(crate) fn strip(&self, chars: Option<&Sym>, left: bool, right: bool) -> SymResult<Sym> {
        let text: &str = &self.text;
        let Some(chars) = chars else {
            let a = if left {
                text.len() - text.trim_start_matches(py_isspace).len()
            } else {
                0
            };
            let b = if right {
                a + text[a..].trim_end_matches(py_isspace).len()
            } else {
                text.len()
            };
            return Ok(self.slice_bytes(a, b.max(a)));
        };
        let set: Vec<(char, Kind)> = chars.char_kinds();
        let plain = !((self.kinds & BIT_C != 0 && chars.kinds & BIT_DU != 0)
            || (self.kinds & BIT_DU != 0 && chars.kinds & BIT_C != 0));
        if plain {
            let cs: Vec<char> = set.iter().map(|(c, _)| *c).collect();
            let a = if left {
                text.len() - text.trim_start_matches(&cs[..]).len()
            } else {
                0
            };
            let b = if right {
                a + text[a..].trim_end_matches(&cs[..]).len()
            } else {
                text.len()
            };
            return Ok(self.slice_bytes(a, b.max(a)));
        }
        // client text never strips a control token's characters, nor the other way round
        let kinds = self.char_kinds();
        let strippable = |i: usize| -> SymResult<bool> {
            let (c, k) = kinds[i];
            for &(sc, sk) in set.iter().filter(|(sc, _)| *sc == c) {
                let _ = sc;
                let ck = matches!(k, Kind::Control(_));
                let sck = matches!(sk, Kind::Control(_));
                if ck != sck {
                    let other = if ck { sk } else { k };
                    if other == Kind::Data {
                        continue;
                    }
                    if other == Kind::Unknown {
                        return Err(SymError::Unknown(
                            "nemotron4 structural rendering: strip lines up text of unknown origin \
                             with a control token"
                                .into(),
                        ));
                    }
                }
                return Ok(true);
            }
            Ok(false)
        };
        let (mut a, mut b) = (0, kinds.len());
        while left && a < b && strippable(a)? {
            a += 1;
        }
        while right && b > a && strippable(b - 1)? {
            b -= 1;
        }
        Ok(self.slice_chars(a, b))
    }

    /// `fill * left + self + fill * right`; `fill` is template text unless a `Sym` is given.
    pub(crate) fn pad(&self, left: usize, right: usize, fill: &Sym) -> Sym {
        let mut b = SymBuilder::new();
        for _ in 0..left {
            b.push_sym(fill);
        }
        b.push_sym(self);
        for _ in 0..right {
            b.push_sym(fill);
        }
        b.finish()
    }

    /// A character mapping of the whole value (case mappings), each run's part of the result where
    /// its characters went; a control token keeps its id only if its spelling is unchanged.
    pub(crate) fn map_chars(&self, f: impl Fn(&str, Option<char>) -> String) -> Sym {
        // f(run text, the character before the run) maps one run's text in its context
        let mut b = SymBuilder::new();
        let mut prev: Option<char> = None;
        for (kind, text) in self.runs() {
            let mapped = f(text, prev);
            let kind = if mapped == text { kind } else { kind.cut() };
            b.push(kind, &mapped);
            prev = text.chars().last().or(prev);
        }
        b.finish()
    }

    pub(crate) fn lower(&self) -> Sym {
        self.map_chars(|t, _| py_lower(t))
    }

    pub(crate) fn upper(&self) -> Sym {
        self.map_chars(|t, _| t.to_uppercase())
    }

    pub(crate) fn swapcase(&self) -> Sym {
        self.map_chars(|t, _| {
            t.chars()
                .flat_map(|c| -> Vec<char> {
                    if c.is_uppercase() {
                        c.to_lowercase().collect()
                    } else if c.is_lowercase() {
                        c.to_uppercase().collect()
                    } else {
                        vec![c]
                    }
                })
                .collect()
        })
    }

    /// Python's `str.title`: a cased character after an uncased one is uppercased (titlecased),
    /// other cased characters lowercased.
    pub(crate) fn title(&self) -> Sym {
        self.map_chars(|t, prev| {
            let mut out = String::with_capacity(t.len());
            let mut prev_cased = prev.is_some_and(is_cased);
            for c in t.chars() {
                if is_cased(c) {
                    if prev_cased {
                        out.extend(c.to_lowercase());
                    } else {
                        out.extend(c.to_uppercase());
                    }
                    prev_cased = true;
                } else {
                    out.push(c);
                    prev_cased = false;
                }
            }
            out
        })
    }

    /// Python's `str.capitalize`: the first character uppercased (titlecased), the rest lowercased.
    pub(crate) fn capitalize(&self) -> Sym {
        self.map_chars(|t, prev| {
            let mut out = String::with_capacity(t.len());
            for (i, c) in t.chars().enumerate() {
                if i == 0 && prev.is_none() {
                    out.extend(c.to_uppercase());
                } else {
                    out.extend(c.to_lowercase());
                }
            }
            out
        })
    }
}

/// Python's `str.lower` (Rust's, which also maps a final sigma by its context, as Python's does).
fn py_lower(text: &str) -> String {
    text.to_lowercase()
}

fn is_cased(c: char) -> bool {
    c.is_uppercase() || c.is_lowercase()
}

fn next_char_len(text: &str, p: usize) -> usize {
    text[p..].chars().next().map_or(1, char::len_utf8)
}

fn prev_char_len(text: &str, p: usize) -> usize {
    text[..p].chars().next_back().map_or(1, char::len_utf8)
}

/// Python's slice bounds for a sequence of `n` items (`slice.indices`).
pub(crate) fn slice_bounds(n: i64, start: Option<i64>, stop: Option<i64>, step: i64) -> (i64, i64) {
    let (lo, hi) = if step < 0 { (-1, n - 1) } else { (0, n) };
    let fix = |v: Option<i64>, default: i64| match v {
        None => default,
        Some(v) if v < 0 => (v + n).max(lo),
        Some(v) => v.min(hi),
    };
    let start = fix(start, if step < 0 { hi } else { lo });
    let stop = fix(stop, if step < 0 { lo } else { hi });
    (start, stop)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctl(id: u32, s: &str) -> Sym {
        Sym::of(Kind::Control(id), s)
    }

    fn cat(parts: &[Sym]) -> Sym {
        Sym::empty().join(parts)
    }

    #[test]
    fn client_text_never_equals_a_control_token() {
        let think = ctl(12, "<think>");
        let data = Sym::data("<think>");
        let tmpl = Sym::template("<think>");
        assert!(!data.py_eq(&think).unwrap());
        assert!(!think.py_eq(&data).unwrap());
        assert!(think.py_eq(&tmpl).unwrap()); // the template's own text is read as its token
        assert!(data.py_eq(&Sym::template("<think>")).unwrap());
        let s = cat(&[
            Sym::data("a <think> b "),
            think.clone(),
            Sym::template(" c"),
        ]);
        assert_eq!(s.find(&think, None, None).unwrap(), 12);
        assert_eq!(s.find(&Sym::data("<think>"), None, None).unwrap(), 2);
        assert_eq!(s.count(&think, None, None).unwrap(), 1);
        let parts = s.split(Some(&think), -1).unwrap();
        assert_eq!(
            parts.iter().map(Sym::as_str).collect::<Vec<_>>(),
            ["a <think> b ", " c"]
        );
        assert!(Sym::unknown("<think>").py_eq(&think).is_err());
    }

    #[test]
    fn slicing_cuts_a_control_token_into_template_text() {
        let s = cat(&[Sym::data("x"), ctl(1, "<c>"), Sym::data("y")]);
        let cut = s.slice_chars(1, 3);
        assert_eq!(cut.as_str(), "<c");
        assert_eq!(cut.runs().collect::<Vec<_>>(), [(Kind::Template, "<c")]);
        let whole = s.slice_chars(1, 4);
        assert_eq!(
            whole.runs().collect::<Vec<_>>(),
            [(Kind::Control(1), "<c>")]
        );
        assert_eq!(s.slice(None, None, Some(-1)).unwrap().as_str(), "y>c<x");
    }

    #[test]
    fn python_string_methods() {
        let s = Sym::data("  a b\tc  ");
        assert_eq!(
            s.split(None, -1).unwrap().iter().map(Sym::as_str).collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert_eq!(
            s.split(None, 1).unwrap().iter().map(Sym::as_str).collect::<Vec<_>>(),
            ["a", "b\tc  "]
        );
        assert_eq!(
            s.rsplit(None, 1).unwrap().iter().map(Sym::as_str).collect::<Vec<_>>(),
            ["a b", "c"]
        );
        assert_eq!(s.strip(None, true, true).unwrap().as_str(), "a b\tc");
        assert_eq!(
            Sym::data("xxhixx")
                .strip(Some(&Sym::template("x")), true, false)
                .unwrap()
                .as_str(),
            "hixx"
        );
        assert_eq!(
            Sym::data("a\nb\r\nc")
                .splitlines(false)
                .iter()
                .map(Sym::as_str)
                .collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert_eq!(
            Sym::data("aaa")
                .replace(&Sym::template("a"), &Sym::template("b"), 2)
                .unwrap()
                .as_str(),
            "bba"
        );
        assert_eq!(
            Sym::data("ab")
                .replace(&Sym::template(""), &Sym::template("-"), -1)
                .unwrap()
                .as_str(),
            "-a-b-"
        );
        assert_eq!(Sym::data("hello world's").title().as_str(), "Hello World'S");
        assert_eq!(Sym::data("hELLO").capitalize().as_str(), "Hello");
        assert_eq!(
            Sym::data("éa").find(&Sym::template("a"), None, None).unwrap(),
            1
        );
        assert_eq!(
            Sym::data("a,b,c")
                .rsplit(Some(&Sym::template(",")), 1)
                .unwrap()
                .iter()
                .map(Sym::as_str)
                .collect::<Vec<_>>(),
            ["a,b", "c"]
        );
        assert_eq!(
            Sym::data("abcabc").rfind(&Sym::template("bc"), None, None).unwrap(),
            4
        );
    }
}
