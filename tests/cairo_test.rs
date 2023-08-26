#![allow(non_snake_case)]

use ark_std::{end_timer, start_timer};
use halo2_base::gates::builder::{
    CircuitBuilderStage, GateThreadBuilder, MultiPhaseThreadBreakPoints, RangeCircuitBuilder,
};
use halo2_base::halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use halo2_base::utils::biguint_to_fe;
use halo2_base::utils::modulus;
use halo2_base::utils::ScalarField;
use halo2_base::Context;
use halo2_cairo::cairo::CairoChip;
use halo2_cairo::cairo::CairoVM;
use num_bigint::BigUint;
const DEGREE: u32 = 11;
const MAX_NUM_CYCLES: usize = 40;

fn state_transition_test<F: ScalarField>(
    ctx: &mut Context<F>,
    pc: F,
    ap: F,
    fp: F,
    memory: Vec<F>,
    expected_next_pc: F,
    expected_next_ap: F,
    expected_next_fp: F,
) {
    let cairo_state = CairoChip::<F, 10>::new();

    let [pc, ap, fp] = [pc, ap, fp].map(|x| ctx.load_witness(x));
    let memory = ctx.assign_witnesses(memory);

    let (next_pc, next_ap, next_fp) = cairo_state.state_transition(ctx, &memory, pc, ap, fp);
    assert_eq!(*next_pc.value(), expected_next_pc);
    assert_eq!(*next_ap.value(), expected_next_ap);
    assert_eq!(*next_fp.value(), expected_next_fp);
}

fn vm_test<F: ScalarField>(
    ctx: &mut Context<F>,
    pc: F,
    ap: F,
    fp: F,
    num_cycles: F,
    memory: Vec<F>,
    expected_final_pc: F,
    expected_final_ap: F,
    expected_final_fp: F,
) {
    let cairo_chip = CairoChip::<F, MAX_NUM_CYCLES>::new();

    let [pc, ap, fp] = [pc, ap, fp].map(|x| ctx.load_witness(x));
    let memory = ctx.assign_witnesses(memory);

    let (final_pc, final_ap, final_fp, is_valid_transition) =
        cairo_chip.vm(ctx, &memory, pc, ap, fp, num_cycles);

    // check register matches
    assert_eq!(*final_pc.value(), expected_final_pc);
    assert_eq!(*final_ap.value(), expected_final_ap);
    assert_eq!(*final_fp.value(), expected_final_fp);

    // check execution trace is valid
    assert_eq!(*is_valid_transition.value(), F::one());
}

fn generate_test_data_for_fibonacci_program<F: ScalarField>() -> (Vec<F>, Vec<[F; 3]>) {
    // [0x48307ffe7fff8000, 0x010780017fff7fff, -1, 1, 2, 3]
    // section 3.1 of cairo paper
    let memory = vec![
        "5201798300658794496",
        "74168662805676031",
        "21888242871839275222246405745257275088548364400416034343698204186575808495616",
        "1",
        "1",
        "2",
        "3",
    ]
    .into_iter()
    .map(|x| F::from_str_vartime(x).unwrap())
    .collect();

    let register_traces = vec![
        [F::from(0u64), F::from(5u64), F::from(5u64)],
        [F::from(1u64), F::from(6u64), F::from(5u64)],
        [F::from(0u64), F::from(6u64), F::from(5u64)],
        [F::from(1u64), F::from(7u64), F::from(5u64)],
    ];

    (memory, register_traces)
}

fn generate_test_data_for_program_with_builtin<F: ScalarField>() -> (Vec<F>, Vec<[F; 3]>) {
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

    (memory, register_traces)
}

fn generate_test_data_for_fibonacci_program_with_wrong_registers<F: ScalarField>(
) -> (Vec<F>, Vec<[F; 3]>) {
    // [0x48307ffe7fff8000, 0x010780017fff7fff, -1, 1, 2, 3]
    // section 3.1 of cairo paper
    let memory = vec![
        "5201798300658794496",
        "74168662805676031",
        "21888242871839275222246405745257275088548364400416034343698204186575808495616",
        "1",
        "1",
        "2",
        "3",
    ]
    .into_iter()
    .map(|x| F::from_str_vartime(x).unwrap())
    .collect();

    let register_traces = vec![
        [F::from(0u64), F::from(5u64), F::from(5u64)],
        [F::from(1u64), F::from(6u64), F::from(5u64)],
        [F::from(0u64), F::from(6u64), F::from(5u64)],
        [F::from(0u64), F::from(0u64), F::from(0u64)],
    ];

    (memory, register_traces)
}

fn state_transition_circuit(
    pc: Fr,
    ap: Fr,
    fp: Fr,
    memory: Vec<Fr>,
    expected_next_pc: Fr,
    expected_next_ap: Fr,
    expected_next_fp: Fr,
    stage: CircuitBuilderStage,
    break_points: Option<MultiPhaseThreadBreakPoints>,
) -> RangeCircuitBuilder<Fr> {
    let mut builder = match stage {
        CircuitBuilderStage::Mock => GateThreadBuilder::mock(),
        CircuitBuilderStage::Prover => GateThreadBuilder::prover(),
        CircuitBuilderStage::Keygen => GateThreadBuilder::keygen(),
    };
    let start0 = start_timer!(|| format!("Witness generation for circuit in {stage:?} stage"));
    state_transition_test(
        builder.main(0),
        pc,
        ap,
        fp,
        memory,
        expected_next_pc,
        expected_next_ap,
        expected_next_fp,
    );

    let circuit = match stage {
        CircuitBuilderStage::Mock => {
            builder.config(DEGREE as usize, Some(20));
            RangeCircuitBuilder::mock(builder)
        }
        CircuitBuilderStage::Keygen => {
            builder.config(DEGREE as usize, Some(20));
            RangeCircuitBuilder::keygen(builder)
        }
        CircuitBuilderStage::Prover => RangeCircuitBuilder::prover(builder, break_points.unwrap()),
    };
    end_timer!(start0);
    circuit
}

fn vm_circuit(
    pc: Fr,
    ap: Fr,
    fp: Fr,
    num_cycles: Fr,
    memory: Vec<Fr>,
    expected_final_pc: Fr,
    expected_final_ap: Fr,
    expected_final_fp: Fr,
    stage: CircuitBuilderStage,
    break_points: Option<MultiPhaseThreadBreakPoints>,
) -> RangeCircuitBuilder<Fr> {
    let mut builder = match stage {
        CircuitBuilderStage::Mock => GateThreadBuilder::mock(),
        CircuitBuilderStage::Prover => GateThreadBuilder::prover(),
        CircuitBuilderStage::Keygen => GateThreadBuilder::keygen(),
    };
    let start0 = start_timer!(|| format!("Witness generation for circuit in {stage:?} stage"));
    vm_test(
        builder.main(0),
        pc,
        ap,
        fp,
        num_cycles,
        memory,
        expected_final_pc,
        expected_final_ap,
        expected_final_fp,
    );

    let circuit = match stage {
        CircuitBuilderStage::Mock => {
            builder.config(DEGREE as usize, Some(20));
            RangeCircuitBuilder::mock(builder)
        }
        CircuitBuilderStage::Keygen => {
            builder.config(DEGREE as usize, Some(20));
            RangeCircuitBuilder::keygen(builder)
        }
        CircuitBuilderStage::Prover => RangeCircuitBuilder::prover(builder, break_points.unwrap()),
    };
    end_timer!(start0);
    circuit
}

#[test]
fn test_cairo_fibonacci_program() {
    let (memory, register_traces) = generate_test_data_for_fibonacci_program::<Fr>();
    for i in 0..register_traces.len() - 1 {
        let (pc, ap, fp) = (
            register_traces[i][0],
            register_traces[i][1],
            register_traces[i][2],
        );
        let (next_pc, next_ap, next_fp) = (
            register_traces[i + 1][0],
            register_traces[i + 1][1],
            register_traces[i + 1][2],
        );
        let circuit = state_transition_circuit(
            pc,
            ap,
            fp,
            memory.clone(),
            next_pc,
            next_ap,
            next_fp,
            CircuitBuilderStage::Mock,
            None,
        );

        // equality contraints hold
        MockProver::run(DEGREE, &circuit, vec![])
            .unwrap()
            .assert_satisfied();
    }
}

#[test]
fn test_cairo_vm_program_with_builtin() {
    let (memory, register_traces) = generate_test_data_for_program_with_builtin::<Fr>();
    let (pc, ap, fp) = (
        register_traces[0][0],
        register_traces[0][1],
        register_traces[0][2],
    );
    let (final_pc, final_ap, final_fp) = (
        register_traces[1][0],
        register_traces[1][1],
        register_traces[1][2],
    );
    let circuit = vm_circuit(
        pc,
        ap,
        fp,
        Fr::from(20), // on cairo playground number of steps is 21
        memory.clone(),
        final_pc,
        final_ap,
        final_fp,
        CircuitBuilderStage::Mock,
        None,
    );

    // equality contraints hold
    MockProver::run(DEGREE, &circuit, vec![])
        .unwrap()
        .assert_satisfied();
}

#[test]
#[should_panic(expected = "assertion failed: `(left == right)`")]
fn test_cairo_fibonacci_program_with_wrong_registers() {
    let (memory, register_traces) =
        generate_test_data_for_fibonacci_program_with_wrong_registers::<Fr>();
    for i in 0..register_traces.len() - 1 {
        let (pc, ap, fp) = (
            register_traces[i][0],
            register_traces[i][1],
            register_traces[i][2],
        );
        let (next_pc, next_ap, next_fp) = (
            register_traces[i + 1][0],
            register_traces[i + 1][1],
            register_traces[i + 1][2],
        );
        let circuit = state_transition_circuit(
            pc,
            ap,
            fp,
            memory.clone(),
            next_pc,
            next_ap,
            next_fp,
            CircuitBuilderStage::Mock,
            None,
        );

        // equality contraints hold
        MockProver::run(DEGREE, &circuit, vec![])
            .unwrap()
            .assert_satisfied();
    }
}
