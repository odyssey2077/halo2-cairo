#![allow(incomplete_features)]

pub mod cairo;

use halo2_base::utils::biguint_to_fe;
use halo2_base::utils::modulus;
use halo2_base::utils::ScalarField;
use num_bigint::BigUint;

pub const MAX_NUM_CYCLES: usize = 40;

pub fn generate_test_data_for_program_with_builtin<F: ScalarField>() -> (Vec<F>, Vec<[F; 3]>, usize)
{
    // This is a test for a longer program, involving builtins, imports and outputs
    // One can generate more tests here: https://www.cairo-lang.org/playground/
    /*
    %builtins output
    from starkware.cairo.common.serialize import serialize_word
    func main{output_ptr : felt*}():
        tempvar x = 10
        tempvar y = x + x
        tempvar z = y * y + x
        serialize_word(x)
        serialize_word(y)
        serialize_word(z)
        return ()
    end
    */
    let memory = vec![
        F::from(0), // dumb entry
        F::from(0x400380007ffc7ffd),
        F::from(0x482680017ffc8000),
        F::from(1),
        F::from(0x208b7fff7fff7ffe),
        F::from(0x480680017fff8000),
        F::from(10),
        F::from(0x48307fff7fff8000),
        F::from(0x48507fff7fff8000),
        F::from(0x48307ffd7fff8000),
        F::from(0x480a7ffd7fff8000),
        F::from(0x48127ffb7fff8000),
        F::from(0x1104800180018000),
        biguint_to_fe(&(modulus::<F>() - BigUint::from(11u16))),
        F::from(0x48127ff87fff8000),
        F::from(0x1104800180018000),
        biguint_to_fe(&(modulus::<F>() - BigUint::from(14u16))),
        F::from(0x48127ff67fff8000),
        F::from(0x1104800180018000),
        biguint_to_fe(&(modulus::<F>() - BigUint::from(17u16))),
        F::from(0x208b7fff7fff7ffe),
        F::from(41), // beginning of outputs
        F::from(44), // end of outputs
        F::from(44), // input
        F::from(10),
        F::from(20),
        F::from(400),
        F::from(410),
        F::from(41),
        F::from(10),
        F::from(24),
        F::from(14),
        F::from(42),
        F::from(20),
        F::from(24),
        F::from(17),
        F::from(43),
        F::from(410),
        F::from(24),
        F::from(20),
        F::from(44),
        F::from(10), // output starts here
        F::from(20),
        F::from(410),
    ];

    let register_traces = vec![
        [F::from(5u64), F::from(24u64), F::from(24u64)],
        [F::from(20u64), F::from(41u64), F::from(24u64)],
    ];

    let num_cycles = 1usize;

    (memory, register_traces, num_cycles)
}
