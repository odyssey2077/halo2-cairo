use ark_std::{start_timer, end_timer};
use halo2_base::{gates::builder::{CircuitBuilderStage, MultiPhaseThreadBreakPoints, RangeCircuitBuilder, GateThreadBuilder}, halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr}, utils::{ScalarField, biguint_to_fe, modulus}, Context};
use halo2_cairo::cairo::{CairoChip, CairoVM};
use num_bigint::BigUint;
use num_traits::Num;
use serde::Deserialize;
use std::{fs::read_to_string, collections::BTreeMap, cmp::Ordering, str::FromStr};

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

#[derive(Debug, Deserialize)]
struct Res {
    memory: BTreeMap<String, String>,
    trace: Vec<BTreeMap<String, u64>>,
    output: String,
}

fn extract_steps(s: &str) -> Option<u64> {
    let reg = regex::Regex::new(r".*steps: ([0-9]+)\n.*").unwrap();
    reg.captures(s).and_then(|c| c.get(1).or(None).map(|h| {
        println!("{:?}", h.as_str());
        return h.as_str().parse::<u64>().unwrap();
}))
}

fn read_data(path: &str) -> (Vec<Fr>, Vec<[Fr; 3]>, u64) {
  let json = format!("{}/res.json", path);
  let contents = read_to_string(json).expect("Should have been able to read the file");
  let res: Res = serde_json::from_str(&contents).expect("Failed to parse JSON");
  let steps = extract_steps(&res.output).expect("extract steps from output");
  let mut sorted_memory_pairs = Vec::from_iter(res.memory.clone());
  sorted_memory_pairs.sort_by(|a, b| {
    let a_int = a.0.parse::<i32>().unwrap();
    let b_int = b.0.parse::<i32>().unwrap();
    if a_int > b_int {
        return Ordering::Greater;
    }
    return Ordering::Less;
  });
  let memory: Vec<String> = sorted_memory_pairs.iter().map(|(_, value)| value.into()).collect();
  let mut memory: Vec<Fr> = memory.iter().map(|m| {
    let m: String = m.into();
    // Cleanup input so that it can be parsed
    if m.starts_with("0x") {
        // Deal with hex numbers
        let m: String = m.strip_prefix("0x").unwrap().to_string();
        return biguint_to_fe(&BigUint::from_str_radix(&m, 16).unwrap());
    } else {
        // Deal with base 10 numbers
        if m.starts_with("-") {
            let m = m.strip_prefix("-").unwrap_or(&m);
            let i = BigUint::from_str(&m).unwrap();
            return biguint_to_fe(&(modulus::<Fr>() - i));
        } else {
            return biguint_to_fe(&BigUint::from_str(&m).unwrap());
        }
    }

  }).collect();
  let trace: Vec<[Fr; 3]> = res.trace.into_iter().map(|m| {
      [Fr::from(*m.get("pc").unwrap()), Fr::from(*m.get("ap").unwrap()), Fr::from(*m.get("fp").unwrap())]
  }).collect();
  memory.insert(0, Fr::from(0));
  return (memory, trace, steps);
}

pub fn run_mock_prover(memory: Vec<Fr>,register_traces: Vec<[Fr; 3]>) {
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

pub fn run_mock_prover_for(path: &str) {
    let (memory, register_traces, steps) = read_data(path);
    let first_trace = register_traces[0];
    let last_trace = register_traces.last().unwrap();
    let (pc, ap, fp) = (
        first_trace[0],
        first_trace[1],
        first_trace[2],
    );
    let (final_pc, final_ap, final_fp) = (
        last_trace[0],
        last_trace[1],
        last_trace[2],
    );
    let circuit = vm_circuit(
      pc,
      ap,
      fp,
      Fr::from(steps),
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
    let num_cycles = ctx.load_witness(num_cycles);

    let (final_pc, final_ap, final_fp, is_valid_transition) =
        cairo_chip.vm(ctx, &memory, pc, ap, fp, num_cycles);

    // check register matches
    assert_eq!(*final_pc.value(), expected_final_pc);
    assert_eq!(*final_ap.value(), expected_final_ap);
    assert_eq!(*final_fp.value(), expected_final_fp);

    // check execution trace is valid
    assert_eq!(*is_valid_transition.value(), F::one());
}

pub fn vm_circuit(
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
            builder.config(DEGREE as usize, None);
            RangeCircuitBuilder::mock(builder)
        }
        CircuitBuilderStage::Keygen => {
            builder.config(DEGREE as usize, None);
            RangeCircuitBuilder::keygen(builder)
        }
        CircuitBuilderStage::Prover => RangeCircuitBuilder::prover(builder, break_points.unwrap()),
    };
    end_timer!(start0);
    circuit
}