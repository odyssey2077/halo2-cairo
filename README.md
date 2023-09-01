# Halo2 Cairo

This repository provides functionabilities to verify cairo program execution using plonkish arithemization + KZG polynomial commitment. This repository is built based on Axiom's halo2-lib library. state transition function is implemented following section 4.5 of Cairo white paper.

## Setup

Install rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Clone this repo:

```bash
git clone https://github.com/odyssey2077/halo2-cairo.git
cd halo2-cairo
```

## Quick start with `halo2-cairo`

### Mock Prover

```bash
LOOKUP_BITS=10 cargo run --example cairo_program_with_builtin -- --name cairo_program_with_builtin -k 11 mock
```

### Key generation

```bash
LOOKUP_BITS=10 cargo run --example cairo_program_with_builtin -- --name cairo_program_with_builtin -k 11 keygen
```

### Proof generation

After you have generated the proving and verifying keys, you can generate a proof using

```bash
LOOKUP_BITS=10 cargo run --example cairo_program_with_builtin -- --name cairo_program_with_builtin -k 11 prove
```

### Proof verification

After you have generated the proving and verifying keys, you can verify a proof using

```bash
LOOKUP_BITS=10 cargo run --example cairo_program_with_builtin -- --name cairo_program_with_builtin -k 11 verify
```
