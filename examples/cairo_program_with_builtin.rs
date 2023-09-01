use clap::Parser;
use halo2_base::utils::ScalarField;
use halo2_base::AssignedValue;
use halo2_base::Context;
use halo2_cairo::cairo::CairoChip;
use halo2_cairo::cairo::CairoVM;
use halo2_cairo::generate_test_data_for_program_with_builtin;
use halo2_cairo::MAX_NUM_CYCLES;
use halo2_scaffold::scaffold::cmd::Cli;
use halo2_scaffold::scaffold::run;
use serde::{Deserialize, Serialize};

// fake input for interface compatibility
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    x: String,
}

fn cairo_vm<F: ScalarField>(
    ctx: &mut Context<F>,
    _: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) {
    let cairo_chip = CairoChip::<F, MAX_NUM_CYCLES>::new();

    let (memory, register_traces, num_cycles) = generate_test_data_for_program_with_builtin::<F>();
    let (pc, ap, fp) = (
        register_traces[0][0],
        register_traces[0][1],
        register_traces[0][2],
    );
    let (expected_final_pc, expected_final_ap, expected_final_fp) = (
        register_traces[1][0],
        register_traces[1][1],
        register_traces[1][2],
    );
    let [pc, ap, fp] = [pc, ap, fp].map(|x| ctx.load_witness(x));
    let memory = ctx.assign_witnesses(memory);
    let num_cycles = ctx.load_witness(F::from(num_cycles as u64));

    let (final_pc, final_ap, final_fp, is_valid_transition) =
        cairo_chip.vm(ctx, &memory, pc, ap, fp, num_cycles);

    assert_eq!(expected_final_ap, *final_ap.value());
    assert_eq!(expected_final_fp, *final_fp.value());
    assert_eq!(expected_final_pc, *final_pc.value());
    assert_eq!(*is_valid_transition.value(), F::one());

    make_public.append(vec![final_pc, final_ap, final_fp, pc, ap, fp, num_cycles].as_mut());
}

fn main() {
    env_logger::init();

    let args = Cli::parse();

    run(cairo_vm, args);
}
