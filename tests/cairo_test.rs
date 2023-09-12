// #![allow(non_snake_case)]

// use common::vm_circuit;
// use halo2_base::gates::builder::CircuitBuilderStage;
// use halo2_base::halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
// use halo2_base::utils::ScalarField;

// mod common;

// use halo2_cairo::generate_test_data_for_program_with_builtin;
// const DEGREE: u32 = 11;

// fn generate_test_data_for_fibonacci_program<F: ScalarField>() -> (Vec<F>, Vec<[F; 3]>) {
//     // [0x48307ffe7fff8000, 0x010780017fff7fff, -1, 1, 2, 3]
//     // section 3.1 of cairo paper
//     let memory = vec![
//         "5201798300658794496",
//         "74168662805676031",
//         "21888242871839275222246405745257275088548364400416034343698204186575808495616",
//         "1",
//         "1",
//         "2",
//         "3",
//     ]
//     .into_iter()
//     .map(|x| F::from_str_vartime(x).unwrap())
//     .collect();

//     let register_traces = vec![
//         [F::from(0u64), F::from(5u64), F::from(5u64)],
//         [F::from(1u64), F::from(6u64), F::from(5u64)],
//         [F::from(0u64), F::from(6u64), F::from(5u64)],
//         [F::from(1u64), F::from(7u64), F::from(5u64)],
//     ];

//     (memory, register_traces)
// }

// fn generate_test_data_for_fibonacci_program_with_wrong_registers<F: ScalarField>(
// ) -> (Vec<F>, Vec<[F; 3]>) {
//     // [0x48307ffe7fff8000, 0x010780017fff7fff, -1, 1, 2, 3]
//     // section 3.1 of cairo paper
//     let memory = vec![
//         "5201798300658794496",
//         "74168662805676031",
//         "21888242871839275222246405745257275088548364400416034343698204186575808495616",
//         "1",
//         "1",
//         "2",
//         "3",
//     ]
//     .into_iter()
//     .map(|x| F::from_str_vartime(x).unwrap())
//     .collect();

//     let register_traces = vec![
//         [F::from(0u64), F::from(5u64), F::from(5u64)],
//         [F::from(1u64), F::from(6u64), F::from(5u64)],
//         [F::from(0u64), F::from(6u64), F::from(5u64)],
//         [F::from(0u64), F::from(0u64), F::from(0u64)],
//     ];

//     (memory, register_traces)
// }

// #[test]
// fn test_cairo_fibonacci_program() {
//     let (memory, register_traces) = generate_test_data_for_fibonacci_program::<Fr>();
//     common::run_mock_prover(memory, register_traces);
// }

// #[test]
// fn test_cairo_vm_program_with_builtin() {
//     let (memory, register_traces, num_cycles) = generate_test_data_for_program_with_builtin::<Fr>();
//     let (pc, ap, fp) = (
//         register_traces[0][0],
//         register_traces[0][1],
//         register_traces[0][2],
//     );
//     let (final_pc, final_ap, final_fp) = (
//         register_traces[1][0],
//         register_traces[1][1],
//         register_traces[1][2],
//     );
//     let circuit = vm_circuit(
//         pc,
//         ap,
//         fp,
//         Fr::from(num_cycles as u64), // on cairo playground number of steps is 21
//         memory.clone(),
//         final_pc,
//         final_ap,
//         final_fp,
//         CircuitBuilderStage::Mock,
//         None,
//     );

//     // equality contraints hold
//     MockProver::run(DEGREE, &circuit, vec![])
//         .unwrap()
//         .assert_satisfied();
// }

// #[test]
// #[should_panic(expected = "assertion failed: `(left == right)`")]
// fn test_cairo_fibonacci_program_with_wrong_registers() {
//     let (memory, register_traces) =
//         generate_test_data_for_fibonacci_program_with_wrong_registers::<Fr>();
//     common::run_mock_prover(memory, register_traces);
// }
