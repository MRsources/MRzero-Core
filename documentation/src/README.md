# Introduction

mdBook doesn't include any logos in the documentation (as banner or similar).
The favicon can be modified as part of the theme.
![logo](logo.png)

# Building the documentation

Install newest (beta) version of mdBook:
```bash
cargo install --git https://github.com/rust-lang/mdBook.git mdbook
```

Run a live-preview server:
```bash
# in the root directory:
mdbook serve documentation

# or if you navigate to the docs directory:
cd documentation
mdbook serve
```