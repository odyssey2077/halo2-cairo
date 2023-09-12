use halo2_base::gates::{GateInstructions, RangeChip, RangeInstructions};
use halo2_base::utils::{fe_to_biguint, ScalarField};
use halo2_base::QuantumCell;
use halo2_base::{
    AssignedValue, Context, QuantumCell::Constant, QuantumCell::Existing, QuantumCell::Witness,
};
use serde::{Deserialize, Serialize};
use std::env::var;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CairoState {
    pub memory: Vec<String>,
    pub pc: String,
    pub ap: String,
    pub fp: String,
}

#[derive(Clone, Copy, Debug)]
pub struct DecodedInstruction<F: ScalarField> {
    off_dst: AssignedValue<F>,
    off_op0: AssignedValue<F>,
    off_op1: AssignedValue<F>,
    dst_reg: AssignedValue<F>,
    op0_reg: AssignedValue<F>,
    op1_src: AssignedValue<F>,
    res_logic: AssignedValue<F>,
    pc_update: AssignedValue<F>,
    ap_update: AssignedValue<F>,
    op_code: AssignedValue<F>,
}

pub trait CairoVM<F: ScalarField, const MAX_CPU_CYCLES: usize> {
    fn vm(
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        pc: AssignedValue<F>,
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
        num_cycles: AssignedValue<F>,
    ) -> (
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
    );
    fn state_transition(
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        pc: AssignedValue<F>,
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
    ) -> (AssignedValue<F>, AssignedValue<F>, AssignedValue<F>);
    fn soft_state_transition(
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        pc: AssignedValue<F>,
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
    ) -> (
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
    );
}

#[derive(Clone, Debug)]
pub struct CairoChip<F: ScalarField, const MAX_CPU_CYCLES: usize> {
    pub range_chip: RangeChip<F>,
}

impl<F: ScalarField, const MAX_CPU_CYCLES: usize> CairoChip<F, MAX_CPU_CYCLES> {
    pub fn new() -> Self {
        // lookup_bits set to 10
        let lookup_bits = var("LOOKUP_BITS")
            .unwrap_or_else(|_| panic!("LOOKUP_BITS not set"))
            .parse()
            .unwrap();
        Self {
            range_chip: RangeChip::<F>::default(lookup_bits),
        }
    }

    pub fn bit_slice(
        &self,
        ctx: &mut Context<F>,
        bits: &Vec<AssignedValue<F>>,
        start: usize,
        end: usize,
    ) -> AssignedValue<F> {
        self.range_chip.gate().inner_product(
            ctx,
            (&bits[start..end]).to_vec(),
            (0..(end - start)).map(|i| Constant(self.range_chip.gate().pow_of_two[i])),
        )
    }

    // recenter a value within [0, 2^16) to [-2^15, 2^15)
    // since the sub here might overflow, the correctness of our circuit relies on any subsequent operations with bias always give positive result
    // e.g. ap + off_op0 >= 0
    pub fn bias(&self, ctx: &mut Context<F>, input: AssignedValue<F>) -> AssignedValue<F> {
        self.range_chip
            .gate()
            .sub(ctx, input, Constant(F::from(2u64.pow(15u32))))
    }

    pub fn decode_instruction(
        &self,
        ctx: &mut Context<F>,
        instruction: AssignedValue<F>,
    ) -> DecodedInstruction<F> {
        let instruction_bits = self.range_chip.gate().num_to_bits(ctx, instruction, 63);
        let off_dst_raw = self.bit_slice(ctx, &instruction_bits, 0, 16);
        let off_dst = self.bias(ctx, off_dst_raw);
        let off_op0_raw = self.bit_slice(ctx, &instruction_bits, 16, 32);
        let off_op0 = self.bias(ctx, off_op0_raw);
        let off_op1_raw = self.bit_slice(ctx, &instruction_bits, 32, 48);
        let off_op1 = self.bias(ctx, off_op1_raw);
        let dst_reg = instruction_bits[48];
        let op0_reg = instruction_bits[49];
        let op1_src = self.bit_slice(ctx, &instruction_bits, 50, 53);
        let res_logic = self.bit_slice(ctx, &instruction_bits, 53, 55);
        let pc_update = self.bit_slice(ctx, &instruction_bits, 55, 58);
        let ap_update = self.bit_slice(ctx, &instruction_bits, 58, 60);
        let op_code = self.bit_slice(ctx, &instruction_bits, 60, 63);

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

    // todo: read memory through dynamic look up table
    pub fn read_memory(
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        address: AssignedValue<F>,
    ) -> AssignedValue<F> {
        self.range_chip
            .gate()
            .select_from_idx(ctx, memory.iter().copied(), address)
    }

    pub fn compute_op0(
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        op0_reg: AssignedValue<F>, // one bit
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
        off_op0: AssignedValue<F>,
    ) -> AssignedValue<F> {
        let ap_plus_off_op0 = self.range_chip.gate().add(ctx, ap, off_op0);
        let fp_plus_off_op0 = self.range_chip.gate().add(ctx, fp, off_op0);
        println!("ap_plus_off_op0: {:?}", ap_plus_off_op0);
        println!("fp_plus_off_op0: {:?}", fp_plus_off_op0);
        println!("op0_reg: {:?}", op0_reg);
        let op0_0 = self.read_memory(ctx, memory, ap_plus_off_op0);
        let op_0_1 = self.read_memory(ctx, memory, fp_plus_off_op0);
        let op0 = self.range_chip.gate().select(ctx, op_0_1, op0_0, op0_reg);
        println!("op0: {:?}", op0);
        op0
    }

    // todo: is undefined behavior handled properly?
    pub fn compute_op1_and_instruction_size(
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        op1_src: AssignedValue<F>,
        op0: AssignedValue<F>,
        off_op1: AssignedValue<F>,
        fp: AssignedValue<F>,
        ap: AssignedValue<F>,
        pc: AssignedValue<F>,
    ) -> (AssignedValue<F>, AssignedValue<F>) {
        //op1_src != 3
        assert!(fe_to_biguint(op1_src.value()) != 3u64.into());
        assert!(fe_to_biguint(op1_src.value()) <= 4u64.into());

        let op0_off_op1 = self.range_chip.gate().add(ctx, op0, off_op1);
        let pc_off_op1 = self.range_chip.gate().add(ctx, pc, off_op1);
        let fp_off_op1 = self.range_chip.gate().add(ctx, fp, off_op1);
        let ap_off_op1 = self.range_chip.gate().add(ctx, ap, off_op1);

        let op1_values: Vec<QuantumCell<F>> = vec![
            Existing(self.read_memory(ctx, memory, op0_off_op1)),
            Existing(self.read_memory(ctx, memory, pc_off_op1)),
            Existing(self.read_memory(ctx, memory, fp_off_op1)),
            Witness(F::zero()), // undefined behavior
            Existing(self.read_memory(ctx, memory, ap_off_op1)),
        ];
        let instruction_values = vec![
            Constant(F::one()),
            Constant(F::from(2u64)),
            Constant(F::one()),
            Witness(F::zero()), // undefined behavior
            Constant(F::one()),
        ];

        let op1 = self
            .range_chip
            .gate()
            .select_from_idx(ctx, op1_values, op1_src);
        let instruction_size =
            self.range_chip
                .gate()
                .select_from_idx(ctx, instruction_values, op1_src);
        (op1, instruction_size)
    }

    pub fn compute_res(
        &self,
        ctx: &mut Context<F>,
        pc_update: AssignedValue<F>,
        res_logic: AssignedValue<F>,
        op1: AssignedValue<F>,
        op0: AssignedValue<F>,
    ) -> AssignedValue<F> {
        assert!(fe_to_biguint(pc_update.value()) != 3u64.into());
        assert!(fe_to_biguint(pc_update.value()) <= 4u64.into());
        assert!(fe_to_biguint(res_logic.value()) <= 2u64.into());

        let op1_op0 = self.range_chip.gate().add(ctx, op1, op0);
        let op1_mul_op0 = self.range_chip.gate().mul(ctx, op1, op0);
        let case_0_1_2_value = Existing(self.range_chip.gate().select_from_idx(
            ctx,
            vec![op1, op1_op0, op1_mul_op0],
            res_logic,
        ));
        let res_values = [
            case_0_1_2_value,
            case_0_1_2_value,
            case_0_1_2_value,
            Witness(F::zero()), // undefined behavior
            Witness(F::zero()), // undefined behavior
        ];
        let res = self
            .range_chip
            .gate()
            .select_from_idx(ctx, res_values, pc_update);
        res
    }

    pub fn compute_dst(
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
        off_dst: AssignedValue<F>,
        dst_reg: AssignedValue<F>,
    ) -> AssignedValue<F> {
        let is_dst_reg_zero = self.range_chip.gate().is_zero(ctx, dst_reg);
        let address_a = self.range_chip.gate().add(ctx, ap, off_dst);
        let var_a = self.read_memory(ctx, memory, address_a);
        let address_b = self.range_chip.gate().add(ctx, fp, off_dst);
        let var_b = self.read_memory(ctx, memory, address_b);
        let dst = self
            .range_chip
            .gate()
            .select(ctx, var_a, var_b, is_dst_reg_zero);
        dst
    }

    pub fn compute_next_pc(
        &self,
        ctx: &mut Context<F>,
        pc: AssignedValue<F>,
        instruction_size: AssignedValue<F>,
        res: AssignedValue<F>,
        dst: AssignedValue<F>,
        op1: AssignedValue<F>,
        pc_update: AssignedValue<F>,
    ) -> AssignedValue<F> {
        assert!(fe_to_biguint(pc_update.value()) != 3u64.into());
        assert!(fe_to_biguint(pc_update.value()) <= 4u64.into());

        let var_a = self.range_chip.gate().add(ctx, pc, instruction_size);
        let var_b = self.range_chip.gate().add(ctx, pc, op1);
        let sel = self.range_chip.gate().is_zero(ctx, dst);
        let case_4_value = self.range_chip.gate().select(ctx, var_a, var_b, sel);
        let next_pc_values = vec![
            Existing(self.range_chip.gate().add(ctx, pc, instruction_size)),
            Existing(res),
            Existing(self.range_chip.gate().add(ctx, pc, res)),
            Witness(F::zero()), // undefined behavior
            Existing(case_4_value),
        ];
        let next_pc = self
            .range_chip
            .gate()
            .select_from_idx(ctx, next_pc_values, pc_update);
        next_pc
    }

    pub fn compute_next_ap_fp(
        &self,
        ctx: &mut Context<F>,
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
        assert!(fe_to_biguint(ap_update.value()) <= 2u64.into());
        assert!(fe_to_biguint(op_code.value()) <= 4u64.into());
        assert!(fe_to_biguint(op_code.value()) != 3u64.into());
        // first, implement assertions
        // if opcode == 1, op0 == pc + instruction_size
        let mut condition = self
            .range_chip
            .gate()
            .is_equal(ctx, op_code, Constant(F::one()));
        let sub_b = self.range_chip.gate().add(ctx, pc, instruction_size);
        let mul_b = self.range_chip.gate().sub(ctx, op0, sub_b);
        let value_to_check_1 = self.range_chip.gate().mul(ctx, condition, mul_b);
        self.range_chip
            .gate()
            .assert_is_const(ctx, &value_to_check_1, &F::zero());
        // if opcode == 1, dst == fp
        let mul_b_2 = self.range_chip.gate().sub(ctx, dst, fp);
        let value_to_check_2 = self.range_chip.gate().mul(ctx, condition, mul_b_2);
        self.range_chip
            .gate()
            .assert_is_const(ctx, &value_to_check_2, &F::zero());

        // if opcode == 4, res = dst
        condition = self
            .range_chip
            .gate()
            .is_equal(ctx, op_code, Constant(F::from(4u64)));
        let mul_b_3 = self.range_chip.gate().sub(ctx, res, dst);
        let value_to_check_3 = self.range_chip.gate().mul(ctx, condition, mul_b_3);
        self.range_chip
            .gate()
            .assert_is_const(ctx, &value_to_check_3, &F::zero());

        // compute next_ap
        let next_ap_value_1 = self.range_chip.gate().add(ctx, ap, res);
        let next_ap_value_2 = self.range_chip.gate().add(ctx, ap, Constant(F::one()));
        let next_ap_swtich_by_ap_update_0_2_4 = self.range_chip.gate().select_from_idx(
            ctx,
            vec![ap, next_ap_value_1, next_ap_value_2],
            ap_update,
        );
        let var_a = self.range_chip.gate().add(ctx, ap, Constant(F::from(2u64)));
        let sel = self.range_chip.gate().is_zero(ctx, ap_update);
        let next_ap_swtich_by_ap_update_1 = self.range_chip.gate().select(
            ctx,
            var_a,
            Witness(F::zero()), // undefined behavior
            sel,
        );
        let next_ap_values = [
            Existing(next_ap_swtich_by_ap_update_0_2_4),
            Existing(next_ap_swtich_by_ap_update_1),
            Existing(next_ap_swtich_by_ap_update_0_2_4),
            Witness(F::zero()), // undefined behavior
            Existing(next_ap_swtich_by_ap_update_0_2_4),
        ];
        let next_ap = self
            .range_chip
            .gate()
            .select_from_idx(ctx, next_ap_values, op_code);

        // compute next_fp
        let next_fp_values = [
            Existing(fp),
            Existing(self.range_chip.gate().add(ctx, ap, Constant(F::from(2u64)))),
            Existing(dst),
            Witness(F::zero()),
            Existing(fp),
        ];
        let next_fp = self
            .range_chip
            .gate()
            .select_from_idx(ctx, next_fp_values, op_code);

        (next_ap, next_fp)
    }

    // returns a boolean to detect if computation is valid instead of panic
    pub fn soft_compute_next_ap_fp(
        &self,
        ctx: &mut Context<F>,
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
        assert!(fe_to_biguint(ap_update.value()) <= 2u64.into());
        assert!(fe_to_biguint(op_code.value()) <= 4u64.into());
        assert!(fe_to_biguint(op_code.value()) != 3u64.into());
        // first, implement assertions
        // if opcode == 1, op0 == pc + instruction_size
        let mut condition = self
            .range_chip
            .gate()
            .is_equal(ctx, op_code, Constant(F::one()));
        let sub_b = self.range_chip.gate().add(ctx, pc, instruction_size);
        let mul_b = self.range_chip.gate().sub(ctx, op0, sub_b);
        let value_to_check_1 = self.range_chip.gate().mul(ctx, condition, mul_b);
        let is_valid_transition_1 =
            self.range_chip
                .gate()
                .is_equal(ctx, value_to_check_1, Constant(F::zero()));
        // if opcode == 1, dst == fp
        let mul_b_2 = self.range_chip.gate().sub(ctx, dst, fp);
        let value_to_check_2 = self.range_chip.gate().mul(ctx, condition, mul_b_2);
        let is_valid_transition_2 =
            self.range_chip
                .gate()
                .is_equal(ctx, value_to_check_2, Constant(F::zero()));

        // if opcode == 4, res = dst
        condition = self
            .range_chip
            .gate()
            .is_equal(ctx, op_code, Constant(F::from(4u64)));
        let mul_b_3 = self.range_chip.gate().sub(ctx, res, dst);
        let value_to_check_3 = self.range_chip.gate().mul(ctx, condition, mul_b_3);
        let is_valid_transition_3 =
            self.range_chip
                .gate()
                .is_equal(ctx, value_to_check_3, Constant(F::zero()));

        // compute next_ap
        let next_ap_value_1 = self.range_chip.gate().add(ctx, ap, res);
        let next_ap_value_2 = self.range_chip.gate().add(ctx, ap, Constant(F::one()));
        let next_ap_swtich_by_ap_update_0_2_4 = self.range_chip.gate().select_from_idx(
            ctx,
            vec![ap, next_ap_value_1, next_ap_value_2],
            ap_update,
        );
        let var_a = self.range_chip.gate().add(ctx, ap, Constant(F::from(2u64)));
        let sel = self.range_chip.gate().is_zero(ctx, ap_update);
        let next_ap_swtich_by_ap_update_1 = self.range_chip.gate().select(
            ctx,
            var_a,
            Witness(F::zero()), // undefined behavior
            sel,
        );
        let next_ap_values = [
            Existing(next_ap_swtich_by_ap_update_0_2_4),
            Existing(next_ap_swtich_by_ap_update_1),
            Existing(next_ap_swtich_by_ap_update_0_2_4),
            Witness(F::zero()), // undefined behavior
            Existing(next_ap_swtich_by_ap_update_0_2_4),
        ];
        let next_ap = self
            .range_chip
            .gate()
            .select_from_idx(ctx, next_ap_values, op_code);

        // compute next_fp
        let next_fp_values = [
            Existing(fp),
            Existing(self.range_chip.gate().add(ctx, ap, Constant(F::from(2u64)))),
            Existing(dst),
            Witness(F::zero()),
            Existing(fp),
        ];
        let next_fp = self
            .range_chip
            .gate()
            .select_from_idx(ctx, next_fp_values, op_code);

        // compute if is valid transition
        let mut is_valid_transition =
            self.range_chip
                .gate()
                .and(ctx, is_valid_transition_1, is_valid_transition_2);
        is_valid_transition =
            self.range_chip
                .gate()
                .and(ctx, is_valid_transition, is_valid_transition_3);

        (next_ap, next_fp, is_valid_transition)
    }
}

impl<F: ScalarField, const MAX_CPU_CYCLES: usize> CairoVM<F, MAX_CPU_CYCLES>
    for CairoChip<F, MAX_CPU_CYCLES>
{
    fn state_transition(
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        pc: AssignedValue<F>,
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
    ) -> (AssignedValue<F>, AssignedValue<F>, AssignedValue<F>) {
        let instruction = self
            .range_chip
            .gate()
            .select_from_idx(ctx, memory.iter().copied(), pc);
        let decoded_instruction = self.decode_instruction(ctx, instruction);
        let op0 = self.compute_op0(
            ctx,
            memory,
            decoded_instruction.op0_reg,
            ap,
            fp,
            decoded_instruction.off_op0,
        );
        let (op1, instruction_size) = self.compute_op1_and_instruction_size(
            ctx,
            memory,
            decoded_instruction.op1_src,
            op0,
            decoded_instruction.off_op1,
            fp,
            ap,
            pc,
        );
        let res = self.compute_res(
            ctx,
            decoded_instruction.pc_update,
            decoded_instruction.res_logic,
            op1,
            op0,
        );
        let dst = self.compute_dst(
            ctx,
            memory,
            ap,
            fp,
            decoded_instruction.off_dst,
            decoded_instruction.dst_reg,
        );
        let next_pc = self.compute_next_pc(
            ctx,
            pc,
            instruction_size,
            res,
            dst,
            op1,
            decoded_instruction.pc_update,
        );
        let (next_ap, next_fp) = self.compute_next_ap_fp(
            ctx,
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
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        pc: AssignedValue<F>,
        ap: AssignedValue<F>,
        fp: AssignedValue<F>,
    ) -> (
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
    ) {
        let instruction = self
            .range_chip
            .gate()
            .select_from_idx(ctx, memory.iter().copied(), pc);
        let decoded_instruction = self.decode_instruction(ctx, instruction);
        println!("decoded_instruction: {:?}", decoded_instruction);
        let op0 = self.compute_op0(
            ctx,
            memory,
            decoded_instruction.op0_reg,
            ap,
            fp,
            decoded_instruction.off_op0,
        );
        println!("op0: {:?}", op0.value());
        let (op1, instruction_size) = self.compute_op1_and_instruction_size(
            ctx,
            memory,
            decoded_instruction.op1_src,
            op0,
            decoded_instruction.off_op1,
            fp,
            ap,
            pc,
        );
        let res = self.compute_res(
            ctx,
            decoded_instruction.pc_update,
            decoded_instruction.res_logic,
            op1,
            op0,
        );
        let dst = self.compute_dst(
            ctx,
            memory,
            ap,
            fp,
            decoded_instruction.off_dst,
            decoded_instruction.dst_reg,
        );
        let next_pc = self.compute_next_pc(
            ctx,
            pc,
            instruction_size,
            res,
            dst,
            op1,
            decoded_instruction.pc_update,
        );
        let (next_ap, next_fp, is_valid_transition) = self.soft_compute_next_ap_fp(
            ctx,
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
        &self,
        ctx: &mut Context<F>,
        memory: &[AssignedValue<F>],
        mut pc: AssignedValue<F>,
        mut ap: AssignedValue<F>,
        mut fp: AssignedValue<F>,
        num_cycles: AssignedValue<F>,
    ) -> (
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
        AssignedValue<F>,
    ) {
        let mut is_valid_transition = ctx.load_constant(F::one());
        for i in 0..MAX_CPU_CYCLES {
            let current_step = Constant(self.range_chip.gate().get_field_element(i as u64));
            // assume MAX_CPU_CYCLES < 2^10
            let is_within_steps = self
                .range_chip
                .is_less_than(ctx, current_step, num_cycles, 10);
            let (next_pc, next_ap, next_fp, is_current_valid_transition) =
                self.soft_state_transition(ctx, &memory, pc, ap, fp);
            pc = self
                .range_chip
                .gate()
                .select(ctx, next_pc, pc, is_within_steps);
            ap = self
                .range_chip
                .gate()
                .select(ctx, next_ap, ap, is_within_steps);
            fp = self
                .range_chip
                .gate()
                .select(ctx, next_fp, fp, is_within_steps);
            let is_valid_transition_within_steps =
                self.range_chip
                    .gate()
                    .and(ctx, is_valid_transition, is_current_valid_transition);
            is_valid_transition = self.range_chip.gate().select(
                ctx,
                is_valid_transition_within_steps,
                is_valid_transition,
                is_within_steps,
            );
        }
        (pc, ap, fp, is_valid_transition)
    }
}
