use crate::cairo::{CairoState, CairoVM, DecodedInstruction};
use crate::generate_test_data_for_program_with_builtin;
use halo2_base::gates::{GateChip, GateInstructions};
use halo2_base::utils::fe_to_biguint;
use halo2_base::Context;
use halo2_base::QuantumCell;
use halo2_base::QuantumCell::{Constant, Existing, Witness};
use halo2_base::{
    gates::flex_gate::{threads::SinglePhaseCoreManager, FlexGateConfig, FlexGateConfigParams},
    utils::{
        fs::gen_srs,
        halo2::raw_assign_advice,
        testing::{check_proof, gen_proof},
        ScalarField,
    },
    virtual_region::{lookups::LookupAnyManager, manager::VirtualRegionManager},
};
use halo2_base::{
    halo2_proofs::{
        arithmetic::Field,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{keygen_pk, keygen_vk, Advice, Circuit, Column, ConstraintSystem, Error},
        poly::Rotation,
    },
    AssignedValue,
};

#[derive(Clone, Debug)]
struct CairoVMConfig<F: ScalarField> {
    cpu: FlexGateConfig<F>,
    copy: Vec<[Column<Advice>; 2]>,
    // dynamic lookup table
    memory: [Column<Advice>; 2],
}

#[derive(Clone, Default)]
struct CairoVMConfigParams {
    cpu: FlexGateConfigParams,
    copy_columns: usize,
}

struct CairoVMCircuit<F: ScalarField, const CYCLES: usize> {
    // private memory input
    memory: Vec<F>,
    gate: GateChip<F>,

    cpu: SinglePhaseCoreManager<F>,
    ram: LookupAnyManager<F, 2>,

    params: CairoVMConfigParams,
}

impl<F: ScalarField, const CYCLES: usize> CairoVMCircuit<F, CYCLES> {
    fn new(memory: Vec<F>, params: CairoVMConfigParams, witness_gen_only: bool) -> Self {
        let cpu = SinglePhaseCoreManager::new(witness_gen_only, Default::default());
        let ram = LookupAnyManager::new(witness_gen_only, cpu.copy_manager.clone());
        let gate = GateChip::default();
        Self {
            memory,
            gate,
            cpu,
            ram,
            params,
        }
    }

    fn read_memory(&mut self, index: AssignedValue<F>) -> AssignedValue<F> {
        let ctx = self.cpu.main();
        // todo: support memory with size larger than 2^8
        let index_usize = index.value().to_repr().as_ref()[0] as usize;
        println!("index: {}", index_usize);
        let value = self.memory[index_usize];
        let value = ctx.load_witness(value);
        self.ram
            .add_lookup((ctx.type_id(), ctx.id()), [index, value]);
        value
    }

    pub fn bit_slice(
        &mut self,
        bits: &Vec<AssignedValue<F>>,
        start: usize,
        end: usize,
    ) -> AssignedValue<F> {
        let ctx = self.cpu.main();
        self.gate.inner_product(
            ctx,
            (&bits[start..end]).to_vec(),
            (0..(end - start)).map(|i| Constant(self.gate.pow_of_two[i])),
        )
    }

    // recenter a value within [0, 2^16) to [-2^15, 2^15)
    // since the sub here might overflow, the correctness of our circuit relies on any subsequent operations with bias always give positive result
    // e.g. ap + off_op0 >= 0
    pub fn bias(&mut self, input: AssignedValue<F>) -> AssignedValue<F> {
        let ctx = self.cpu.main();
        self.gate
            .sub(ctx, input, Constant(F::from(2u64.pow(15u32))))
    }

    pub fn decode_instruction(&mut self, instruction: AssignedValue<F>) -> DecodedInstruction<F> {
        let ctx = self.cpu.main();
        let instruction_bits = self.gate.num_to_bits(ctx, instruction, 63);
        let off_dst_raw = self.bit_slice(&instruction_bits, 0, 16);
        let off_dst = self.bias(off_dst_raw);
        let off_op0_raw = self.bit_slice(&instruction_bits, 16, 32);
        let off_op0 = self.bias(off_op0_raw);
        let off_op1_raw = self.bit_slice(&instruction_bits, 32, 48);
        let off_op1 = self.bias(off_op1_raw);
        let dst_reg = instruction_bits[48];
        let op0_reg = instruction_bits[49];
        let op1_src = self.bit_slice(&instruction_bits, 50, 53);
        let res_logic = self.bit_slice(&instruction_bits, 53, 55);
        let pc_update = self.bit_slice(&instruction_bits, 55, 58);
        let ap_update = self.bit_slice(&instruction_bits, 58, 60);
        let op_code = self.bit_slice(&instruction_bits, 60, 63);

        DecodedInstruction {
            off_dst,
            off_op0,
            off_op1,
            dst_reg,
            op0_reg,
            op1_src,
            res_logic,
            pc_update,
            ap_update,
            op_code,
        }
    }

    pub fn compute_op0(
        &mut self,
        op0_reg: AssignedValue<F>, // one bit
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
        off_op0: AssignedValue<F>,
    ) -> AssignedValue<F> {
        let mut ctx = self.cpu.main();
        let ap_plus_off_op0 = self.gate.add(ctx, ap, off_op0);
        let fp_plus_off_op0 = self.gate.add(ctx, fp, off_op0);
        let op0_0 = self.read_memory(ap_plus_off_op0);
        let op_0_1 = self.read_memory(fp_plus_off_op0);
        ctx = self.cpu.main();
        let op0 = self.gate.select(ctx, op_0_1, op0_0, op0_reg);
        op0
    }

    // todo: is undefined behavior handled properly?
    pub fn compute_op1_and_instruction_size(
        &mut self,
        op1_src: AssignedValue<F>,
        op0: AssignedValue<F>,
        off_op1: AssignedValue<F>,
        fp: AssignedValue<F>,
        ap: AssignedValue<F>,
        pc: AssignedValue<F>,
    ) -> (AssignedValue<F>, AssignedValue<F>) {
        let mut ctx = self.cpu.main();

        //op1_src != 3
        assert!(fe_to_biguint(op1_src.value()) != 3u64.into());
        assert!(fe_to_biguint(op1_src.value()) <= 4u64.into());

        let op0_off_op1 = self.gate.add(ctx, op0, off_op1);
        let pc_off_op1 = self.gate.add(ctx, pc, off_op1);
        let fp_off_op1 = self.gate.add(ctx, fp, off_op1);
        let ap_off_op1 = self.gate.add(ctx, ap, off_op1);

        println!("op0_off_op1: {:?}", op0_off_op1.value());
        let op1_values: Vec<QuantumCell<F>> = vec![
            Existing(self.read_memory(op0_off_op1)),
            Existing(self.read_memory(pc_off_op1)),
            Existing(self.read_memory(fp_off_op1)),
            Witness(F::ZERO), // undefined behavior
            Existing(self.read_memory(ap_off_op1)),
        ];
        let instruction_values = vec![
            Constant(F::ONE),
            Constant(F::from(2u64)),
            Constant(F::ONE),
            Witness(F::ZERO), // undefined behavior
            Constant(F::ONE),
        ];

        ctx = self.cpu.main();
        let op1 = self.gate.select_from_idx(ctx, op1_values, op1_src);
        let instruction_size = self.gate.select_from_idx(ctx, instruction_values, op1_src);
        (op1, instruction_size)
    }

    pub fn compute_res(
        &mut self,
        pc_update: AssignedValue<F>,
        res_logic: AssignedValue<F>,
        op1: AssignedValue<F>,
        op0: AssignedValue<F>,
    ) -> AssignedValue<F> {
        let ctx = self.cpu.main();

        assert!(fe_to_biguint(pc_update.value()) != 3u64.into());
        assert!(fe_to_biguint(pc_update.value()) <= 4u64.into());
        assert!(fe_to_biguint(res_logic.value()) <= 2u64.into());

        let op1_op0 = self.gate.add(ctx, op1, op0);
        let op1_mul_op0 = self.gate.mul(ctx, op1, op0);
        let case_0_1_2_value = Existing(self.gate.select_from_idx(
            ctx,
            vec![op1, op1_op0, op1_mul_op0],
            res_logic,
        ));
        let res_values = [
            case_0_1_2_value,
            case_0_1_2_value,
            case_0_1_2_value,
            Witness(F::ZERO), // undefined behavior
            Witness(F::ZERO), // undefined behavior
        ];
        let res = self.gate.select_from_idx(ctx, res_values, pc_update);
        res
    }

    pub fn compute_dst(
        &mut self,
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
        off_dst: AssignedValue<F>,
        dst_reg: AssignedValue<F>,
    ) -> AssignedValue<F> {
        let mut ctx = self.cpu.main();
        let is_dst_reg_zero = self.gate.is_zero(ctx, dst_reg);
        let address_a = self.gate.add(ctx, ap, off_dst);
        let address_b = self.gate.add(ctx, fp, off_dst);
        let var_a = self.read_memory(address_a);
        let var_b = self.read_memory(address_b);
        ctx = self.cpu.main();
        let dst = self.gate.select(ctx, var_a, var_b, is_dst_reg_zero);
        dst
    }

    pub fn compute_next_pc(
        &mut self,
        pc: AssignedValue<F>,
        instruction_size: AssignedValue<F>,
        res: AssignedValue<F>,
        dst: AssignedValue<F>,
        op1: AssignedValue<F>,
        pc_update: AssignedValue<F>,
    ) -> AssignedValue<F> {
        let ctx = self.cpu.main();

        assert!(fe_to_biguint(pc_update.value()) != 3u64.into());
        assert!(fe_to_biguint(pc_update.value()) <= 4u64.into());

        let var_a = self.gate.add(ctx, pc, instruction_size);
        let var_b = self.gate.add(ctx, pc, op1);
        let sel = self.gate.is_zero(ctx, dst);
        let case_4_value = self.gate.select(ctx, var_a, var_b, sel);
        let next_pc_values = vec![
            Existing(self.gate.add(ctx, pc, instruction_size)),
            Existing(res),
            Existing(self.gate.add(ctx, pc, res)),
            Witness(F::ZERO), // undefined behavior
            Existing(case_4_value),
        ];
        let next_pc = self.gate.select_from_idx(ctx, next_pc_values, pc_update);
        next_pc
    }

    pub fn compute_next_ap_fp(
        &mut self,
        op_code: AssignedValue<F>,
        pc: AssignedValue<F>,
        instruction_size: AssignedValue<F>,
        res: AssignedValue<F>,
        dst: AssignedValue<F>,
        op0: AssignedValue<F>,
        fp: AssignedValue<F>,
        ap: AssignedValue<F>,
        ap_update: AssignedValue<F>,
    ) -> (AssignedValue<F>, AssignedValue<F>) {
        let ctx = self.cpu.main();

        assert!(fe_to_biguint(ap_update.value()) <= 2u64.into());
        assert!(fe_to_biguint(op_code.value()) <= 4u64.into());
        assert!(fe_to_biguint(op_code.value()) != 3u64.into());
        // first, implement assertions
        // if opcode == 1, op0 == pc + instruction_size
        let mut condition = self.gate.is_equal(ctx, op_code, Constant(F::ONE));
        let sub_b = self.gate.add(ctx, pc, instruction_size);
        let mul_b = self.gate.sub(ctx, op0, sub_b);
        let value_to_check_1 = self.gate.mul(ctx, condition, mul_b);
        self.gate.assert_is_const(ctx, &value_to_check_1, &F::ZERO);
        // if opcode == 1, dst == fp
        let mul_b_2 = self.gate.sub(ctx, dst, fp);
        let value_to_check_2 = self.gate.mul(ctx, condition, mul_b_2);
        self.gate.assert_is_const(ctx, &value_to_check_2, &F::ZERO);

        // if opcode == 4, res = dst
        condition = self.gate.is_equal(ctx, op_code, Constant(F::from(4u64)));
        let mul_b_3 = self.gate.sub(ctx, res, dst);
        let value_to_check_3 = self.gate.mul(ctx, condition, mul_b_3);
        self.gate.assert_is_const(ctx, &value_to_check_3, &F::ZERO);

        // compute next_ap
        let next_ap_value_1 = self.gate.add(ctx, ap, res);
        let next_ap_value_2 = self.gate.add(ctx, ap, Constant(F::ONE));
        let next_ap_swtich_by_ap_update_0_2_4 =
            self.gate
                .select_from_idx(ctx, vec![ap, next_ap_value_1, next_ap_value_2], ap_update);
        let var_a = self.gate.add(ctx, ap, Constant(F::from(2u64)));
        let sel = self.gate.is_zero(ctx, ap_update);
        let next_ap_swtich_by_ap_update_1 = self.gate.select(
            ctx,
            var_a,
            Witness(F::ZERO), // undefined behavior
            sel,
        );
        let next_ap_values = [
            Existing(next_ap_swtich_by_ap_update_0_2_4),
            Existing(next_ap_swtich_by_ap_update_1),
            Existing(next_ap_swtich_by_ap_update_0_2_4),
            Witness(F::ZERO), // undefined behavior
            Existing(next_ap_swtich_by_ap_update_0_2_4),
        ];
        let next_ap = self.gate.select_from_idx(ctx, next_ap_values, op_code);

        // compute next_fp
        let next_fp_values = [
            Existing(fp),
            Existing(self.gate.add(ctx, ap, Constant(F::from(2u64)))),
            Existing(dst),
            Witness(F::ZERO),
            Existing(fp),
        ];
        let next_fp = self.gate.select_from_idx(ctx, next_fp_values, op_code);

        (next_ap, next_fp)
    }

    // returns a boolean to detect if computation is valid instead of panic
    pub fn soft_compute_next_ap_fp(
        &mut self,
        op_code: AssignedValue<F>,
        pc: AssignedValue<F>,
        instruction_size: AssignedValue<F>,
        res: AssignedValue<F>,
        dst: AssignedValue<F>,
        op0: AssignedValue<F>,
        fp: AssignedValue<F>,
        ap: AssignedValue<F>,
        ap_update: AssignedValue<F>,
    ) -> (AssignedValue<F>, AssignedValue<F>, AssignedValue<F>) {
        let ctx = self.cpu.main();

        assert!(fe_to_biguint(ap_update.value()) <= 2u64.into());
        assert!(fe_to_biguint(op_code.value()) <= 4u64.into());
        assert!(fe_to_biguint(op_code.value()) != 3u64.into());
        // first, implement assertions
        // if opcode == 1, op0 == pc + instruction_size
        let mut condition = self.gate.is_equal(ctx, op_code, Constant(F::ONE));
        let sub_b = self.gate.add(ctx, pc, instruction_size);
        let mul_b = self.gate.sub(ctx, op0, sub_b);
        let value_to_check_1 = self.gate.mul(ctx, condition, mul_b);
        let is_valid_transition_1 = self.gate.is_equal(ctx, value_to_check_1, Constant(F::ZERO));
        // if opcode == 1, dst == fp
        let mul_b_2 = self.gate.sub(ctx, dst, fp);
        let value_to_check_2 = self.gate.mul(ctx, condition, mul_b_2);
        let is_valid_transition_2 = self.gate.is_equal(ctx, value_to_check_2, Constant(F::ZERO));

        // if opcode == 4, res = dst
        condition = self.gate.is_equal(ctx, op_code, Constant(F::from(4u64)));
        let mul_b_3 = self.gate.sub(ctx, res, dst);
        let value_to_check_3 = self.gate.mul(ctx, condition, mul_b_3);
        let is_valid_transition_3 = self.gate.is_equal(ctx, value_to_check_3, Constant(F::ZERO));

        // compute next_ap
        let next_ap_value_1 = self.gate.add(ctx, ap, res);
        let next_ap_value_2 = self.gate.add(ctx, ap, Constant(F::ONE));
        let next_ap_swtich_by_ap_update_0_2_4 =
            self.gate
                .select_from_idx(ctx, vec![ap, next_ap_value_1, next_ap_value_2], ap_update);
        let var_a = self.gate.add(ctx, ap, Constant(F::from(2u64)));
        let sel = self.gate.is_zero(ctx, ap_update);
        let next_ap_swtich_by_ap_update_1 = self.gate.select(
            ctx,
            var_a,
            Witness(F::ZERO), // undefined behavior
            sel,
        );
        let next_ap_values = [
            Existing(next_ap_swtich_by_ap_update_0_2_4),
            Existing(next_ap_swtich_by_ap_update_1),
            Existing(next_ap_swtich_by_ap_update_0_2_4),
            Witness(F::ZERO), // undefined behavior
            Existing(next_ap_swtich_by_ap_update_0_2_4),
        ];
        let next_ap = self.gate.select_from_idx(ctx, next_ap_values, op_code);

        // compute next_fp
        let next_fp_values = [
            Existing(fp),
            Existing(self.gate.add(ctx, ap, Constant(F::from(2u64)))),
            Existing(dst),
            Witness(F::ZERO),
            Existing(fp),
        ];
        let next_fp = self.gate.select_from_idx(ctx, next_fp_values, op_code);

        // compute if is valid transition
        let mut is_valid_transition =
            self.gate
                .and(ctx, is_valid_transition_1, is_valid_transition_2);
        is_valid_transition = self
            .gate
            .and(ctx, is_valid_transition, is_valid_transition_3);

        (next_ap, next_fp, is_valid_transition)
    }
}

impl<F: ScalarField, const MAX_CPU_CYCLES: usize> CairoVM<F, MAX_CPU_CYCLES>
    for CairoVMCircuit<F, MAX_CPU_CYCLES>
{
    fn state_transition(
        &mut self,
        pc: AssignedValue<F>,
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
    ) -> (AssignedValue<F>, AssignedValue<F>, AssignedValue<F>) {
        let instruction = self.read_memory(pc);
        let decoded_instruction = self.decode_instruction(instruction);
        let op0 = self.compute_op0(
            decoded_instruction.op0_reg,
            ap,
            fp,
            decoded_instruction.off_op0,
        );
        let (op1, instruction_size) = self.compute_op1_and_instruction_size(
            decoded_instruction.op1_src,
            op0,
            decoded_instruction.off_op1,
            fp,
            ap,
            pc,
        );
        let res = self.compute_res(
            decoded_instruction.pc_update,
            decoded_instruction.res_logic,
            op1,
            op0,
        );
        let dst = self.compute_dst(
            ap,
            fp,
            decoded_instruction.off_dst,
            decoded_instruction.dst_reg,
        );
        let next_pc = self.compute_next_pc(
            pc,
            instruction_size,
            res,
            dst,
            op1,
            decoded_instruction.pc_update,
        );
        let (next_ap, next_fp) = self.compute_next_ap_fp(
            decoded_instruction.op_code,
            pc,
            instruction_size,
            res,
            dst,
            op0,
            fp,
            ap,
            decoded_instruction.ap_update,
        );
        (next_pc, next_ap, next_fp)
    }

    fn soft_state_transition(
        &mut self,
        pc: AssignedValue<F>,
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
    ) -> (
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
    ) {
        let instruction = self.read_memory(pc);
        let decoded_instruction = self.decode_instruction(instruction);
        let op0 = self.compute_op0(
            decoded_instruction.op0_reg,
            ap,
            fp,
            decoded_instruction.off_op0,
        );
        let (op1, instruction_size) = self.compute_op1_and_instruction_size(
            decoded_instruction.op1_src,
            op0,
            decoded_instruction.off_op1,
            fp,
            ap,
            pc,
        );
        let res = self.compute_res(
            decoded_instruction.pc_update,
            decoded_instruction.res_logic,
            op1,
            op0,
        );
        let dst = self.compute_dst(
            ap,
            fp,
            decoded_instruction.off_dst,
            decoded_instruction.dst_reg,
        );
        let next_pc = self.compute_next_pc(
            pc,
            instruction_size,
            res,
            dst,
            op1,
            decoded_instruction.pc_update,
        );
        let (next_ap, next_fp, is_valid_transition) = self.soft_compute_next_ap_fp(
            decoded_instruction.op_code,
            pc,
            instruction_size,
            res,
            dst,
            op0,
            fp,
            ap,
            decoded_instruction.ap_update,
        );
        (next_pc, next_ap, next_fp, is_valid_transition)
    }

    // return (pc, ap, fp) after the vm execute for num_cycles and a boolean to indicate if the execution trace is valid
    fn vm(
        &mut self,
        mut pc: AssignedValue<F>,
        mut ap: AssignedValue<F>,
        mut fp: AssignedValue<F>,
        // _num_cycles: AssignedValue<F>,
    ) -> (
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
    ) {
        let ctx = self.cpu.main();

        let mut is_valid_transition = ctx.load_constant(F::ONE);
        for _ in 0..MAX_CPU_CYCLES {
            // let current_step = Constant(self.gate.get_field_element(i as u64));
            // assume MAX_CPU_CYCLES < 2^10
            // let is_within_steps = self
            //     .range_chip
            //     .is_less_than(ctx, current_step, num_cycles, 10);
            // integrate range chip later
            let (next_pc, next_ap, next_fp, is_current_valid_transition) =
                self.soft_state_transition(pc, ap, fp);
            pc = next_pc;
            ap = next_ap;
            fp = next_fp;
            let ctx = self.cpu.main();
            is_valid_transition =
                self.gate
                    .and(ctx, is_valid_transition, is_current_valid_transition);
        }
        (pc, ap, fp, is_valid_transition)
    }
}

impl<F: ScalarField, const CYCLES: usize> Circuit<F> for CairoVMCircuit<F, CYCLES> {
    type Config = CairoVMConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = CairoVMConfigParams;

    fn params(&self) -> Self::Params {
        self.params.clone()
    }

    fn without_witnesses(&self) -> Self {
        unimplemented!()
    }

    fn configure_with_params(meta: &mut ConstraintSystem<F>, params: Self::Params) -> Self::Config {
        let k = params.cpu.k;
        let mut cpu = FlexGateConfig::configure(meta, params.cpu);
        let copy: Vec<_> = (0..params.copy_columns)
            .map(|_| {
                [(); 2].map(|_| {
                    let advice = meta.advice_column();
                    meta.enable_equality(advice);
                    advice
                })
            })
            .collect();
        let mem = [meta.advice_column(), meta.advice_column()];

        for copy in &copy {
            meta.lookup_any("dynamic memory lookup table", |meta| {
                let mem = mem.map(|c| meta.query_advice(c, Rotation::cur()));
                let copy = copy.map(|c| meta.query_advice(c, Rotation::cur()));
                vec![
                    (copy[0].clone(), mem[0].clone()),
                    (copy[1].clone(), mem[1].clone()),
                ]
            });
        }
        cpu.max_rows = (1 << k) - meta.minimum_rows();

        CairoVMConfig {
            cpu,
            copy,
            memory: mem,
        }
    }

    fn configure(_: &mut ConstraintSystem<F>) -> Self::Config {
        unreachable!()
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "Cairo VM Circuit",
            |mut region| {
                // Raw assign the private memory inputs
                for (i, &value) in self.memory.iter().enumerate() {
                    let idx = Value::known(F::from(i as u64));
                    raw_assign_advice(&mut region, config.memory[0], i, idx);
                    raw_assign_advice(&mut region, config.memory[1], i, Value::known(value));
                }
                self.cpu.assign_raw(
                    &(config.cpu.basic_gates[0].clone(), config.cpu.max_rows),
                    &mut region,
                );
                self.ram.assign_raw(&config.copy, &mut region);
                self.cpu
                    .copy_manager
                    .assign_raw(&config.cpu.constants, &mut region);
                Ok(())
            },
        )
    }
}

#[test]
fn test_cairo_mock() {
    let k = 5u32;
    let (memory, register_traces, num_cycles) = generate_test_data_for_program_with_builtin();
    const NUM_CYCLES: usize = 20;
    let pc: Fr = register_traces[0][0];
    let ap: Fr = register_traces[0][1];
    let fp: Fr = register_traces[0][2];

    let usable_rows = 2usize.pow(k) - 11; // guess
    let copy_columns = num_cycles / usable_rows + 1;
    let params = CairoVMConfigParams::default();
    let mut circuit = CairoVMCircuit::<Fr, NUM_CYCLES>::new(memory, params, false);
    let ctx = circuit.cpu.main();
    let [pc, ap, fp] = [pc, ap, fp].map(|x| ctx.load_witness(x));
    circuit.vm(pc, ap, fp);
    // auto-configuration stuff
    let num_advice = circuit.cpu.total_advice() / usable_rows + 1;
    circuit.params.cpu = FlexGateConfigParams {
        k: k as usize,
        num_advice_per_phase: vec![num_advice],
        num_fixed: 1,
    };
    circuit.params.copy_columns = copy_columns;
    MockProver::run(k, &circuit, vec![])
        .unwrap()
        .assert_satisfied();
}

#[test]
fn test_cairo_prover() {
    let k = 5u32;
    let (memory, register_traces, num_cycles) = generate_test_data_for_program_with_builtin();
    const NUM_CYCLES: usize = 20;
    let pc: Fr = register_traces[0][0];
    let ap: Fr = register_traces[0][1];
    let fp: Fr = register_traces[0][2];
    let usable_rows = 2usize.pow(k) - 11; // guess
    let copy_columns = num_cycles / usable_rows + 1;
    let params = CairoVMConfigParams::default();
    let mut circuit = CairoVMCircuit::<Fr, NUM_CYCLES>::new(memory.clone(), params, false);
    let ctx = circuit.cpu.main();
    let [pc, ap, fp] = [pc, ap, fp].map(|x| ctx.load_witness(x));
    circuit.vm(pc, ap, fp);
    let num_advice = circuit.cpu.total_advice() / usable_rows + 1;
    circuit.params.cpu = FlexGateConfigParams {
        k: k as usize,
        num_advice_per_phase: vec![num_advice],
        num_fixed: 1,
    };
    circuit.params.copy_columns = copy_columns;

    let params = gen_srs(k);
    let vk = keygen_vk(&params, &circuit).unwrap();
    let pk = keygen_pk(&params, vk, &circuit).unwrap();
    let circuit_params = circuit.params();
    let break_points = circuit.cpu.break_points.borrow().clone().unwrap();
    drop(circuit);

    let mut circuit = CairoVMCircuit::<Fr, NUM_CYCLES>::new(memory, circuit_params, true);
    let ctx = circuit.cpu.main();
    let pc: Fr = register_traces[0][0];
    let ap: Fr = register_traces[0][1];
    let fp: Fr = register_traces[0][2];
    let [pc, ap, fp] = [pc, ap, fp].map(|x| ctx.load_witness(x));
    *circuit.cpu.break_points.borrow_mut() = Some(break_points);
    circuit.vm(pc, ap, fp);

    let proof = gen_proof(&params, &pk, circuit);
    check_proof(&params, pk.get_vk(), &proof, true);
}
