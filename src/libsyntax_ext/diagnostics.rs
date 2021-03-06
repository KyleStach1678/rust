// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

// Error messages for EXXXX errors.
// Each message should start and end with a new line, and be wrapped to 80 characters.
// In vim you can `:set tw=80` and use `gq` to wrap paragraphs. Use `:set tw=0` to disable.
register_long_diagnostics! {
E0660: r##"
The argument to the `asm` macro is not well-formed.

Erroneous code example:

```compile_fail,E0660
asm!("nop" "nop");
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,

E0661: r##"
An invalid syntax was passed to the second argument of an `asm` macro line.

Erroneous code example:

```compile_fail,E0661
let a;
asm!("nop" : "r"(a));
```

Considering that this would be a long explanation, we instead recommend you to
take a look at the unstable book:
https://doc.rust-lang.org/unstable-book/language-features/asm.html
"##,
}
