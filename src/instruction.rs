//! Ghidra-style, flat p-code operations and their operands.
//!
//! These types represent one instruction as a sequence of operations over
//! varnodes. They deliberately complement, rather than replace, the
//! source-shaped [`crate::PcodeAst`]. A producer lowers nested expressions to
//! these operations by allocating temporaries in its unique space.

use crate::{
    Ast, AstNode, BinaryOperator, BitRangeFieldId, Builtin, Expression, ExpressionTy, FieldId,
    Ident, LabelOrNode, Load, LocalVarId, PCodeOpId, PcodeAst, Range, RangeParam, RegisterId,
    SPACE_CONST, SpaceId, TableId, UnaryOperator,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fmt,
};

/// A storage location or constant used as an input or output of a p-code op.
///
/// `space` and `offset` identify the location; `size` is its width in bytes.
/// A constant is represented by [`SPACE_CONST`] and stores its value in
/// `offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Varnode {
    /// Address space containing this varnode, or [`SPACE_CONST`] for a constant.
    pub space: SpaceId,
    /// Byte offset in `space`, or the value for a constant-space varnode.
    pub offset: u64,
    /// Width in bytes.
    pub size: usize,
}

impl Varnode {
    /// Creates a varnode at `offset` in `space` with byte width `size`.
    pub const fn new(space: SpaceId, offset: u64, size: usize) -> Self {
        Self {
            space,
            offset,
            size,
        }
    }

    /// Creates a constant-space varnode containing `value` with byte width `size`.
    pub const fn constant(value: u64, size: usize) -> Self {
        Self::new(SPACE_CONST, value, size)
    }

    /// Returns whether this is a constant-space varnode.
    pub fn is_constant(self) -> bool {
        self.space == SPACE_CONST
    }
}

/// An opcode from Ghidra's p-code operation reference.
///
/// The variants use Rust-style names; each maps one-for-one to Ghidra's
/// `CPUI_*` opcode with the corresponding uppercase spelling. Some variants
/// are analysis-only pseudo-operations and are identified by
/// [`Self::is_raw_instruction_op`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Opcode {
    Copy = 1,
    Load,
    Store,
    Branch,
    CBranch,
    BranchInd,
    Call,
    CallInd,
    CallOther,
    Return,
    IntEqual,
    IntNotEqual,
    IntSLess,
    IntSLessEqual,
    IntLess,
    IntLessEqual,
    IntZext,
    IntSext,
    IntAdd,
    IntSub,
    IntCarry,
    IntSCarry,
    IntSBorrow,
    Int2Comp,
    IntNegate,
    IntXor,
    IntAnd,
    IntOr,
    IntLeft,
    IntRight,
    IntSRight,
    IntMult,
    IntDiv,
    IntSDiv,
    IntRem,
    IntSRem,
    BoolNegate,
    BoolXor,
    BoolAnd,
    BoolOr,
    FloatEqual,
    FloatNotEqual,
    FloatLess,
    FloatLessEqual,
    // Ghidra reserves opcode slot 45.
    FloatNan = 46,
    FloatAdd,
    FloatDiv,
    FloatMult,
    FloatSub,
    FloatNeg,
    FloatAbs,
    FloatSqrt,
    FloatInt2Float,
    FloatFloat2Float,
    FloatTrunc,
    FloatCeil,
    FloatFloor,
    FloatRound,
    MultiEqual,
    Indirect,
    Piece,
    SubPiece,
    Cast,
    PtrAdd,
    PtrSub,
    SegmentOp,
    CpoolRef,
    New,
    Insert,
    Extract,
    PopCount,
    LzCount,
}

impl Opcode {
    /// Every opcode in Ghidra's p-code operation reference.
    pub const ALL: &[Self] = &[
        Self::Copy,
        Self::Load,
        Self::Store,
        Self::Branch,
        Self::CBranch,
        Self::BranchInd,
        Self::Call,
        Self::CallInd,
        Self::CallOther,
        Self::Return,
        Self::IntEqual,
        Self::IntNotEqual,
        Self::IntSLess,
        Self::IntSLessEqual,
        Self::IntLess,
        Self::IntLessEqual,
        Self::IntZext,
        Self::IntSext,
        Self::IntAdd,
        Self::IntSub,
        Self::IntCarry,
        Self::IntSCarry,
        Self::IntSBorrow,
        Self::Int2Comp,
        Self::IntNegate,
        Self::IntXor,
        Self::IntAnd,
        Self::IntOr,
        Self::IntLeft,
        Self::IntRight,
        Self::IntSRight,
        Self::IntMult,
        Self::IntDiv,
        Self::IntSDiv,
        Self::IntRem,
        Self::IntSRem,
        Self::BoolNegate,
        Self::BoolXor,
        Self::BoolAnd,
        Self::BoolOr,
        Self::FloatEqual,
        Self::FloatNotEqual,
        Self::FloatLess,
        Self::FloatLessEqual,
        Self::FloatNan,
        Self::FloatAdd,
        Self::FloatDiv,
        Self::FloatMult,
        Self::FloatSub,
        Self::FloatNeg,
        Self::FloatAbs,
        Self::FloatSqrt,
        Self::FloatInt2Float,
        Self::FloatFloat2Float,
        Self::FloatTrunc,
        Self::FloatCeil,
        Self::FloatFloor,
        Self::FloatRound,
        Self::MultiEqual,
        Self::Indirect,
        Self::Piece,
        Self::SubPiece,
        Self::Cast,
        Self::PtrAdd,
        Self::PtrSub,
        Self::SegmentOp,
        Self::CpoolRef,
        Self::New,
        Self::Insert,
        Self::Extract,
        Self::PopCount,
        Self::LzCount,
    ];

    /// Returns Ghidra's numeric `CPUI_*` opcode value.
    pub const fn ghidra_id(self) -> u8 {
        self as u8
    }

    /// Returns whether this opcode may occur in raw instruction p-code.
    ///
    /// Ghidra reserves SSA and type-recovery pseudo-operations for later
    /// analysis. `INSERT` and `EXTRACT` are likewise pseudo-operations even
    /// though they correspond to SLEIGH bit-range syntax.
    pub const fn is_raw_instruction_op(self) -> bool {
        !matches!(
            self,
            Self::MultiEqual
                | Self::Indirect
                | Self::Cast
                | Self::PtrAdd
                | Self::PtrSub
                | Self::SegmentOp
                | Self::Insert
                | Self::Extract
        )
    }
}

/// One flat p-code operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PcodeOp {
    /// The operation to perform.
    pub opcode: Opcode,
    /// Destination varnode, if this operation produces a value.
    pub output: Option<Varnode>,
    /// Source varnodes in Ghidra's documented operand order.
    pub inputs: Vec<Varnode>,
}

impl PcodeOp {
    /// Creates an operation with its output and ordered inputs.
    pub fn new(opcode: Opcode, output: Option<Varnode>, inputs: Vec<Varnode>) -> Self {
        Self {
            opcode,
            output,
            inputs,
        }
    }
}

/// Flat p-code emitted for one machine instruction.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct InstructionPcode {
    /// Operations in execution order.
    pub ops: Vec<PcodeOp>,
}

impl InstructionPcode {
    /// Creates an empty instruction p-code sequence.
    pub const fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Returns whether this instruction has no p-code operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Metadata for a named SLEIGH bit range.
///
/// The lowerer uses this to turn reads and writes of a bit-range identifier
/// into raw p-code operations over its containing varnode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitRangeInfo {
    /// The containing register or other storage varnode.
    pub storage: Varnode,
    /// Index of the least-significant bit in `storage`.
    pub start: usize,
    /// Number of bits in the named range.
    pub size: usize,
}

/// Producer-specific information required to lower a source-shaped AST.
///
/// `wazabin-pcode` owns the lowering algorithm and calls this trait for
/// specification data. A SLEIGH compiler can implement it using its compiled
/// specification without making this crate depend on that compiler.
pub trait PcodeLoweringContext {
    /// The specification's default address space.
    fn default_space(&self) -> SpaceId;
    /// The unique address space used for deterministic temporary varnodes.
    fn unique_space(&self) -> SpaceId;
    /// Returns the storage varnode for a register identifier.
    fn register_varnode(&self, id: RegisterId) -> Option<Varnode>;
    /// Returns metadata for a named register bit range.
    fn bitrange_info(&self, id: BitRangeFieldId) -> Option<BitRangeInfo>;
    /// Returns the byte width of offsets in `space`.
    fn address_size(&self, space: SpaceId) -> Option<usize>;
}

/// Failure while lowering source-shaped SLEIGH p-code to flat operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PcodeLowerError {
    /// An expression needs a byte width but neither the AST nor its consumer supplies one.
    UnknownSize,
    /// A zero-byte varnode was requested.
    ZeroSize,
    /// A raw `COPY` would use different widths for its input and output.
    CopySizeMismatch { input: usize, output: usize },
    /// A raw operation requires equally-sized input varnodes.
    InputSizeMismatch {
        operation: &'static str,
        left: usize,
        right: usize,
    },
    /// A comparison or boolean operation did not produce a one-byte result.
    InvalidBooleanSize(usize),
    /// A memory pointer did not have the address width of its referenced space.
    AddressSizeMismatch { expected: usize, actual: usize },
    /// A store's declared width differed from the value being stored.
    StoreSizeMismatch { declared: usize, value: usize },
    /// A bit range is empty, exceeds its containing varnode, or cannot fit in a u64 mask.
    InvalidRange {
        start: usize,
        size: usize,
        storage_bits: usize,
    },
    /// Allocating a temporary overflowed the unique-space offset.
    UniqueSpaceOverflow,
    /// The lowering context did not know a register referenced by the AST.
    UnknownRegister(RegisterId),
    /// A source-level field, table, or global survived expansion.
    UnresolvedIdentifier(&'static str),
    /// A source-level memory space survived resolution.
    UnresolvedSpace,
    /// A source-level macro parameter survived expansion.
    UnresolvedRangeParameter,
    /// A source construct that must be expanded before lowering survived.
    InternalNode(&'static str),
    /// The operation cannot be represented as raw instruction p-code yet.
    Unsupported(&'static str),
    /// A label was declared more than once in one instruction.
    DuplicateLabel(Box<str>),
    /// A branch referred to a label that was not declared in this instruction.
    UnknownLabel(Box<str>),
    /// A direct branch target was not a literal machine address or local label.
    InvalidDirectTarget,
}

impl fmt::Display for PcodeLowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSize => write!(f, "p-code expression has no known byte width"),
            Self::ZeroSize => write!(f, "p-code varnodes cannot have zero byte width"),
            Self::CopySizeMismatch { input, output } => write!(
                f,
                "p-code COPY requires equal widths, got {input} and {output} bytes"
            ),
            Self::InputSizeMismatch {
                operation,
                left,
                right,
            } => write!(
                f,
                "p-code {operation} requires equal input widths, got {left} and {right} bytes"
            ),
            Self::InvalidBooleanSize(size) => {
                write!(
                    f,
                    "p-code comparison and boolean outputs must be one byte, got {size}"
                )
            }
            Self::AddressSizeMismatch { expected, actual } => write!(
                f,
                "p-code memory pointer requires {expected} bytes, got {actual}"
            ),
            Self::StoreSizeMismatch { declared, value } => write!(
                f,
                "p-code STORE declares {declared} bytes but receives {value}"
            ),
            Self::InvalidRange {
                start,
                size,
                storage_bits,
            } => write!(
                f,
                "invalid bit range [{start}, {size}] for {storage_bits}-bit storage"
            ),
            Self::UniqueSpaceOverflow => write!(f, "unique-space temporary allocation overflowed"),
            Self::UnknownRegister(id) => write!(f, "unknown register {}", usize::from(*id)),
            Self::UnresolvedIdentifier(kind) => {
                write!(f, "unresolved {kind} reached p-code lowering")
            }
            Self::UnresolvedSpace => write!(f, "unresolved address space reached p-code lowering"),
            Self::UnresolvedRangeParameter => {
                write!(f, "unresolved range parameter reached p-code lowering")
            }
            Self::InternalNode(kind) => write!(f, "unexpanded {kind} reached p-code lowering"),
            Self::Unsupported(what) => write!(f, "raw p-code lowering does not support {what}"),
            Self::DuplicateLabel(label) => write!(f, "duplicate p-code label `{label}`"),
            Self::UnknownLabel(label) => write!(f, "unknown p-code label `{label}`"),
            Self::InvalidDirectTarget => write!(f, "invalid direct p-code branch target"),
        }
    }
}

impl std::error::Error for PcodeLowerError {}

/// Identifies one instruction-local p-code label.
///
/// A streaming emitter cannot know the operation index of a forward label, so
/// labels reach a [`PcodeSink`] symbolically. Identifiers are assigned in the
/// order the AST defines the labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LabelId(u32);

impl LabelId {
    /// Returns this label's index, which is stable within one [`PcodePlan`].
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Rebuilds a label from an index returned by [`index`](Self::index).
    pub const fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Whole-instruction facts a consumer needs before p-code emission starts.
///
/// Planning is a read-only pass over the expanded AST. It exists so a consumer
/// can prepare instruction-wide state — out-of-instruction branch and call
/// destinations, instruction-local labels — without a flat p-code vector to
/// re-scan.
#[derive(Debug, Clone, Default)]
pub struct PcodePlan {
    /// Local widths resolved from their uses, needed before a forward-only
    /// emitter can allocate a local's temporary.
    local_sizes: HashMap<LocalVarId, usize>,
    labels: Vec<Box<str>>,
    /// Whether each label stands at the end of the instruction, where it is
    /// the machine instruction's fall-through rather than a local block.
    terminal: Vec<bool>,
    label_ids: HashMap<Box<str>, LabelId>,
    direct_branches: Vec<u64>,
    direct_calls: Vec<u64>,
}

impl PcodePlan {
    /// Names of the instruction-local labels, indexed by [`LabelId::index`].
    pub fn labels(&self) -> &[Box<str>] {
        &self.labels
    }

    /// Returns whether `label` stands after this instruction's last
    /// operation. Such a label is the machine instruction's fall-through: a
    /// consumer should send a branch to it wherever execution continues after
    /// the instruction, rather than open a block for it.
    pub fn is_terminal(&self, label: LabelId) -> bool {
        self.terminal[label.index()]
    }

    /// Addresses this instruction can reach with a direct branch.
    pub fn direct_branches(&self) -> &[u64] {
        &self.direct_branches
    }

    /// Addresses this instruction can reach with a direct call.
    pub fn direct_calls(&self) -> &[u64] {
        &self.direct_calls
    }

    /// Declares an instruction-local label and returns its identifier.
    ///
    /// A name is only assigned one identifier: a duplicate *definition* is an
    /// emission-time error, reported with the name. This is public so a
    /// consumer holding already-flattened p-code can rebuild an equivalent
    /// plan for the same emitter.
    pub fn declare_label(&mut self, label: &str) -> LabelId {
        if let Some(id) = self.label_ids.get(label) {
            return *id;
        }
        let id = LabelId(self.labels.len() as u32);
        self.labels.push(Box::from(label));
        self.terminal.push(false);
        self.label_ids.insert(Box::from(label), id);
        id
    }

    /// Declares an address this instruction can reach with a direct branch.
    pub fn declare_direct_branch(&mut self, address: u64) {
        if !self.direct_branches.contains(&address) {
            self.direct_branches.push(address);
        }
    }

    /// Declares an address this instruction can reach with a direct call.
    pub fn declare_direct_call(&mut self, address: u64) {
        if !self.direct_calls.contains(&address) {
            self.direct_calls.push(address);
        }
    }

    fn label_id(&self, label: &str) -> Option<LabelId> {
        self.label_ids.get(label).copied()
    }
}

/// Receives resolved p-code operations as an instruction is emitted.
///
/// Operations arrive in execution order. Sinks are infallible: a consumer that
/// can fail records its own error and ignores the rest of the instruction,
/// because a partially emitted instruction is discarded by its caller.
pub trait PcodeSink {
    /// Receives one resolved operation. `inputs` is borrowed for the call
    /// only, so a sink which needs to retain the operation must copy it.
    fn op(&mut self, opcode: Opcode, output: Option<Varnode>, inputs: &[Varnode]);

    /// Marks the position of an instruction-local label: the next operation
    /// reported is its target.
    fn label(&mut self, label: LabelId);

    /// Receives a branch whose target is instruction-local. `opcode` is
    /// [`Opcode::Branch`] or [`Opcode::CBranch`], and `condition` is present
    /// exactly for the latter.
    fn branch_label(&mut self, opcode: Opcode, label: LabelId, condition: Option<Varnode>);
}

/// The sink which reproduces [`InstructionPcode`]: it retains operations and
/// resolves local branches into the relative constant targets raw p-code uses.
#[derive(Debug, Default)]
struct Collector {
    ops: Vec<PcodeOp>,
    label_ops: HashMap<LabelId, usize>,
    fixups: Vec<(usize, LabelId)>,
}

impl Collector {
    fn finish(mut self, plan: &PcodePlan) -> Result<Vec<PcodeOp>, PcodeLowerError> {
        for (op_index, label) in &self.fixups {
            let target = *self
                .label_ops
                .get(label)
                .ok_or_else(|| PcodeLowerError::UnknownLabel(plan.labels[label.index()].clone()))?;
            let relative = i64::try_from(target)
                .ok()
                .and_then(|target| {
                    i64::try_from(*op_index)
                        .ok()
                        .and_then(|source| target.checked_sub(source))
                })
                .ok_or(PcodeLowerError::UniqueSpaceOverflow)?;
            self.ops[*op_index].inputs[0] = Varnode::constant(relative as u64, 8);
        }
        Ok(self.ops)
    }
}

impl PcodeSink for Collector {
    fn op(&mut self, opcode: Opcode, output: Option<Varnode>, inputs: &[Varnode]) {
        self.ops.push(PcodeOp::new(opcode, output, inputs.to_vec()));
    }

    fn label(&mut self, label: LabelId) {
        self.label_ops.insert(label, self.ops.len());
    }

    fn branch_label(&mut self, opcode: Opcode, label: LabelId, condition: Option<Varnode>) {
        let mut inputs = Vec::with_capacity(1 + usize::from(condition.is_some()));
        inputs.push(Varnode::constant(0, 8));
        inputs.extend(condition);
        self.fixups.push((self.ops.len(), label));
        self.ops.push(PcodeOp::new(opcode, None, inputs));
    }
}

/// Collects the whole-instruction facts of `ast` without emitting p-code.
///
/// The plan is the shared contract between the p-code emitter and its
/// consumer: it is produced from the same expanded AST that emission lowers,
/// so a consumer never has to re-scan flat p-code to discover them.
pub fn plan_instruction(
    ast: &PcodeAst,
    context: &impl PcodeLoweringContext,
) -> Result<PcodePlan, PcodeLowerError> {
    let mut planner = Planner {
        context,
        plan: PcodePlan::default(),
    };
    planner.plan(ast);
    Ok(planner.plan)
}

/// A width in the domain a width-inference pass works over.
///
/// Per-instruction planning knows every width as a concrete byte count. A
/// producer resolving its own source bodies does not: a body's local can take
/// its width from a table operand whose export only exists once an instruction
/// is decoded. Naming that dependency rather than dropping it is what makes
/// the two passes agree — a pass that simply skipped the unknown would size
/// the local from a *later* statement and reach a different answer.
pub trait Width: Copy + Eq + std::fmt::Debug {
    /// A concrete width in bytes.
    fn fixed(size: usize) -> Self;

    /// This width as a byte count, if it is concrete. Arithmetic on a width
    /// uses this, so a symbolic width yields no constraint rather than a
    /// wrong one.
    fn size(self) -> Option<usize>;

    /// The width of the value an operand supplies, if this domain can name
    /// it. Concrete domains cannot, and report the width as unknown.
    fn operand(_key: OperandKey) -> Option<Self> {
        None
    }
}

impl Width for usize {
    fn fixed(size: usize) -> Self {
        size
    }

    fn size(self) -> Option<usize> {
        Some(self)
    }
}

/// An operand whose value a decode substitutes into a p-code body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperandKey {
    /// A sub-table operand, which supplies its constructor's exported value.
    Table(TableId),
    /// A decoder field, which supplies the constant this encoding gave it.
    Field(FieldId),
}

/// A width resolved before decoding, which may still name an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolicWidth {
    /// A concrete width in bytes.
    Fixed(usize),
    /// The width of the value an operand supplies.
    SameAs(OperandKey),
}

impl Width for SymbolicWidth {
    fn fixed(size: usize) -> Self {
        Self::Fixed(size)
    }

    fn size(self) -> Option<usize> {
        match self {
            Self::Fixed(size) => Some(size),
            Self::SameAs(_) => None,
        }
    }

    fn operand(key: OperandKey) -> Option<Self> {
        Some(Self::SameAs(key))
    }
}

/// Widths, in bytes, of the local variables of one p-code body.
pub type LocalSizes = HashMap<LocalVarId, usize>;

/// Infers the local-variable widths of a *source* p-code body.
///
/// This is the same inference [`plan_instruction`] runs per decoded
/// instruction, exposed so a producer can resolve widths once per body at
/// specification-compile time — reporting an unsizable local as a compile
/// error rather than an `UnknownSize` at lift time, and leaving nothing for
/// the per-instruction planner to iterate.
///
/// A local absent from the result could not be sized from this body alone: its
/// width comes from a value the producer substitutes into the body, so the
/// producer must resolve it or fall back to [`plan_instruction`].
pub fn infer_local_sizes<S, W: Width>(
    statements: &[Ast<S>],
    context: &impl PcodeLoweringContext,
) -> HashMap<LocalVarId, W> {
    SizeInference::run(context, statements)
}

/// Plans `ast` with local widths the producer has already resolved.
///
/// The caller guarantees `local_sizes` covers every local the body uses; a
/// missing width is an `UnknownSize` error at emission, exactly as it is when
/// the per-instruction inference cannot resolve one.
pub fn plan_instruction_with(
    ast: &PcodeAst,
    context: &impl PcodeLoweringContext,
    local_sizes: LocalSizes,
) -> Result<PcodePlan, PcodeLowerError> {
    let mut planner = Planner {
        context,
        plan: PcodePlan {
            local_sizes,
            ..PcodePlan::default()
        },
    };
    planner.plan_statements(ast);
    Ok(planner.plan)
}

/// Lowers `ast` and reports each resolved operation to `sink`.
///
/// `plan` must come from [`plan_instruction`] for the same AST and context.
/// Unlike [`lower_instruction`], no operation vector is built: local branches
/// are reported against the plan's labels rather than resolved offsets.
pub fn emit_instruction(
    ast: &PcodeAst,
    context: &impl PcodeLoweringContext,
    plan: &PcodePlan,
    sink: &mut impl PcodeSink,
) -> Result<(), PcodeLowerError> {
    Lowerer::new(context, plan, sink).emit_all(ast)
}

/// Lowers a fully expanded source-shaped AST to Ghidra-style instruction p-code.
///
/// The AST must be in consumer form: `build`, `export`, delay-slot, macro,
/// deferred-name, field, and table nodes are rejected. The lowerer allocates
/// deterministic temporaries in [`PcodeLoweringContext::unique_space`].
///
/// Bit ranges are expanded into shifts, masks, and `SUBPIECE` rather than
/// Ghidra's analysis-only `INSERT` and `EXTRACT` pseudo-operations. Range
/// writes whose parent storage exceeds 64 bits are rejected because this
/// source AST's literals cannot represent their full clear mask.
pub fn lower_instruction(
    ast: &PcodeAst,
    context: &impl PcodeLoweringContext,
) -> Result<InstructionPcode, PcodeLowerError> {
    Ok(InstructionPcode {
        ops: collect_ops(ast, context)?,
    })
}

fn collect_ops(
    ast: &PcodeAst,
    context: &impl PcodeLoweringContext,
) -> Result<Vec<PcodeOp>, PcodeLowerError> {
    let plan = plan_instruction(ast, context)?;
    let mut collector = Collector::default();
    emit_instruction(ast, context, &plan, &mut collector)?;
    collector.finish(&plan)
}

/// Lowers `ast` and exposes its resolved flat p-code to `sink` without
/// materializing an [`InstructionPcode`] for the caller.
///
/// The lowerer still keeps operations temporarily while resolving local branch
/// labels. Consumers which can process a complete instruction synchronously
/// can therefore avoid making the flat p-code an owned boundary object.
pub fn lower_instruction_into<R>(
    ast: &PcodeAst,
    context: &impl PcodeLoweringContext,
    sink: impl FnOnce(&[PcodeOp]) -> R,
) -> Result<R, PcodeLowerError> {
    let ops = collect_ops(ast, context)?;
    Ok(sink(&ops))
}

impl InstructionPcode {
    /// Lowers `ast` using producer-specific information from `context`.
    pub fn lower(
        ast: &PcodeAst,
        context: &impl PcodeLoweringContext,
    ) -> Result<Self, PcodeLowerError> {
        lower_instruction(ast, context)
    }
}

struct Lowerer<'a, 'p, 's, C: PcodeLoweringContext + ?Sized, S: PcodeSink + ?Sized> {
    context: &'a C,
    plan: &'p PcodePlan,
    sink: &'s mut S,
    locals: HashMap<LocalVarId, Varnode>,
    defined_labels: HashSet<LabelId>,
    next_unique: u64,
}

impl<'a, 'p, 's, C: PcodeLoweringContext + ?Sized, S: PcodeSink + ?Sized>
    Lowerer<'a, 'p, 's, C, S>
{
    fn new(context: &'a C, plan: &'p PcodePlan, sink: &'s mut S) -> Self {
        Self {
            context,
            plan,
            sink,
            locals: HashMap::new(),
            defined_labels: HashSet::new(),
            next_unique: 0,
        }
    }

    fn emit_all(mut self, ast: &PcodeAst) -> Result<(), PcodeLowerError> {
        for statement in &ast.statements {
            self.lower_statement(&statement.ty)?;
        }
        Ok(())
    }

    fn emit(&mut self, opcode: Opcode, output: Option<Varnode>, inputs: &[Varnode]) {
        self.sink.op(opcode, output, inputs);
    }

    fn lower_statement(&mut self, statement: &AstNode) -> Result<(), PcodeLowerError> {
        match statement {
            AstNode::Assignment {
                lhs: Ident::BitRange(id),
                rhs,
                ..
            } => {
                let info = self
                    .context
                    .bitrange_info(*id)
                    .ok_or(PcodeLowerError::Unsupported("an unknown named bit range"))?;
                self.insert_range(info.storage, info.start, info.size, rhs)?;
            }
            AstNode::Assignment { lhs, size, rhs } => {
                let output =
                    self.storage_for_ident(lhs.clone(), size.or_else(|| self.expr_size(rhs)))?;
                self.lower_expr(rhs, Some(output))?;
            }
            AstNode::LoadAssignment { lhs, rhs, .. } => self.lower_store(lhs, rhs)?,
            AstNode::RangeAssignment { lhs, rhs, .. } => self.lower_range_assignment(lhs, rhs)?,
            AstNode::Build(_) => return Err(PcodeLowerError::InternalNode("build statement")),
            AstNode::DelaySlot(_) => {
                return Err(PcodeLowerError::InternalNode("delay-slot directive"));
            }
            AstNode::DeferredBuild(_) => {
                return Err(PcodeLowerError::InternalNode("deferred build statement"));
            }
            AstNode::Label(label) => {
                let id = self.label_id(label)?;
                if !self.defined_labels.insert(id) {
                    return Err(PcodeLowerError::DuplicateLabel(label.clone()));
                }
                self.sink.label(id);
            }
            AstNode::Branch { target } => self.lower_direct_flow(Opcode::Branch, target, None)?,
            AstNode::ConditionalBranch { condition, target } => {
                let condition = self.lower_expr(condition, None)?;
                self.lower_direct_flow(Opcode::CBranch, target, Some(condition))?;
            }
            AstNode::BranchIndirect { target } => {
                self.lower_indirect_flow(Opcode::BranchInd, target)?
            }
            AstNode::Call { target } => self.lower_direct_flow(Opcode::Call, target, None)?,
            AstNode::CallIndirect { target } => {
                self.lower_indirect_flow(Opcode::CallInd, target)?
            }
            AstNode::Return { target } => self.lower_indirect_flow(Opcode::Return, target)?,
            AstNode::Export(_) => return Err(PcodeLowerError::InternalNode("export statement")),
            AstNode::Expression(expr) => self.lower_effect(expr)?,
        }
        Ok(())
    }

    fn lower_store(&mut self, load: &Load, rhs: &Expression) -> Result<(), PcodeLowerError> {
        let space = self.load_space(load)?;
        if space == SPACE_CONST {
            return Err(PcodeLowerError::Unsupported("a store to constant space"));
        }
        let ptr = self.lower_expr(&load.ptr, None)?;
        self.validate_pointer(space, ptr)?;
        let value = match load.size {
            Some(size) => self.lower_expr_with_size(rhs, size)?,
            None => self.lower_expr(rhs, None)?,
        };
        if let Some(size) = load.size {
            Self::checked_size(size)?;
            if size != value.size {
                return Err(PcodeLowerError::StoreSizeMismatch {
                    declared: size,
                    value: value.size,
                });
            }
        }
        self.emit(Opcode::Store, None, &[Self::space_id(space), ptr, value]);
        Ok(())
    }

    fn lower_direct_flow(
        &mut self,
        opcode: Opcode,
        target: &LabelOrNode,
        condition: Option<Varnode>,
    ) -> Result<(), PcodeLowerError> {
        let target = match target {
            LabelOrNode::Label(label) => {
                // A local branch keeps its symbolic target: a streaming sink
                // cannot be handed a relative offset to a label it has not
                // reached yet.
                let id = self.label_id(label)?;
                self.sink.branch_label(opcode, id, condition);
                return Ok(());
            }
            LabelOrNode::Node(_) => {
                return Err(PcodeLowerError::InternalNode("unresolved branch target"));
            }
            LabelOrNode::Expr(expr) => self.direct_target(expr)?,
        };
        match condition {
            Some(condition) => self.emit(opcode, None, &[target, condition]),
            None => self.emit(opcode, None, &[target]),
        }
        Ok(())
    }

    fn lower_indirect_flow(
        &mut self,
        opcode: Opcode,
        target: &Expression,
    ) -> Result<(), PcodeLowerError> {
        let target = self.lower_expr(target, None)?;
        self.emit(opcode, None, &[target]);
        Ok(())
    }

    fn direct_target(&self, target: &Expression) -> Result<Varnode, PcodeLowerError> {
        let ExpressionTy::SizedInt { value, size } = target.ty else {
            return Err(PcodeLowerError::InvalidDirectTarget);
        };
        let size = size
            .or(target.size)
            .or_else(|| self.context.address_size(self.context.default_space()))
            .ok_or(PcodeLowerError::UnknownSize)?;
        Self::checked_size(size)?;
        Ok(Varnode::new(self.context.default_space(), value, size))
    }

    fn lower_effect(&mut self, expr: &Expression) -> Result<(), PcodeLowerError> {
        match &expr.ty {
            ExpressionTy::PcodeOp { id, args } => {
                let inputs = self.lower_userop_inputs(*id, args)?;
                self.emit(Opcode::CallOther, None, &inputs);
                Ok(())
            }
            ExpressionTy::MacroCall { .. } => Err(PcodeLowerError::InternalNode("macro call")),
            ExpressionTy::DeferredCall { .. } => {
                Err(PcodeLowerError::InternalNode("deferred call"))
            }
            _ => Err(PcodeLowerError::Unsupported("a discarded value expression")),
        }
    }

    fn lower_expr(
        &mut self,
        expr: &Expression,
        requested_output: Option<Varnode>,
    ) -> Result<Varnode, PcodeLowerError> {
        match &expr.ty {
            ExpressionTy::SizedInt { value, size } => {
                // Raw p-code integer literals take the width of the
                // operation that consumes them, including an explicitly
                // suffixed source literal used for a wider x86-64 register
                // write (for example `R10 = imm32`).
                let input = Varnode::constant(
                    *value,
                    requested_output
                        .map(|output| output.size)
                        .or(*size)
                        .or(expr.size)
                        .ok_or(PcodeLowerError::UnknownSize)?,
                );
                self.copy_if_requested(input, requested_output)
            }
            ExpressionTy::Ident(Ident::BitRange(id)) => {
                self.lower_named_bitrange(*id, requested_output)
            }
            ExpressionTy::Ident(ident) => {
                let input = self.storage_for_ident(ident.clone(), expr.size)?;
                self.copy_if_requested(input, requested_output)
            }
            ExpressionTy::Load(load) => self.lower_load(expr, load, requested_output),
            ExpressionTy::SubPieceMsb { src, count } => {
                let input = self.lower_expr(src, None)?;
                let size = requested_output
                    .map(|output| output.size)
                    .or(expr.size)
                    .unwrap_or_else(|| input.size.saturating_sub(*count));
                let output = self.output(requested_output, size)?;
                if *count >= input.size || size > input.size - count {
                    return Err(PcodeLowerError::InvalidRange {
                        start: count.saturating_mul(8),
                        size: size.saturating_mul(8),
                        storage_bits: input.size.saturating_mul(8),
                    });
                }
                self.emit(
                    Opcode::SubPiece,
                    Some(output),
                    &[input, Varnode::constant(*count as u64, 8)],
                );
                Ok(output)
            }
            ExpressionTy::SubPieceLsb { src, count } => {
                let input = self.lower_expr(src, None)?;
                if *count == 0 || *count > input.size {
                    return Err(PcodeLowerError::InvalidRange {
                        start: 0,
                        size: count.saturating_mul(8),
                        storage_bits: input.size.saturating_mul(8),
                    });
                }
                let output = self.output(requested_output, *count)?;
                self.emit(
                    Opcode::SubPiece,
                    Some(output),
                    &[input, Varnode::constant(0, 8)],
                );
                Ok(output)
            }
            ExpressionTy::Range(range) => self.lower_range(range, requested_output),
            ExpressionTy::FunctionCall { builtin, args } => {
                self.lower_builtin(expr, *builtin, args, requested_output)
            }
            ExpressionTy::PcodeOp { id, args } => {
                let size = requested_output
                    .map(|output| output.size)
                    .or(expr.size)
                    .ok_or(PcodeLowerError::UnknownSize)?;
                let output = self.output(requested_output, size)?;
                let inputs = self.lower_userop_inputs(*id, args)?;
                self.emit(Opcode::CallOther, Some(output), &inputs);
                Ok(output)
            }
            ExpressionTy::MacroCall { .. } => Err(PcodeLowerError::InternalNode("macro call")),
            ExpressionTy::DeferredCall { .. } => {
                Err(PcodeLowerError::InternalNode("deferred call"))
            }
            ExpressionTy::Unop(unop) => self.lower_unop(expr, unop.op, &unop.e, requested_output),
            ExpressionTy::Binop(binop) => {
                self.lower_binop(expr, binop.op, &binop.lhs, &binop.rhs, requested_output)
            }
        }
    }

    fn lower_load(
        &mut self,
        expr: &Expression,
        load: &Load,
        requested_output: Option<Varnode>,
    ) -> Result<Varnode, PcodeLowerError> {
        let space = self.load_space(load)?;
        let ptr = self.lower_expr(&load.ptr, None)?;
        if space != SPACE_CONST {
            self.validate_pointer(space, ptr)?;
        }
        let size = requested_output
            .map(|output| output.size)
            .or(load.size)
            .or(expr.size)
            .ok_or(PcodeLowerError::UnknownSize)?;
        if space == SPACE_CONST {
            if ptr.size != size {
                return Err(PcodeLowerError::Unsupported(
                    "a constant-space load that changes width",
                ));
            }
            return self.copy_if_requested(ptr, requested_output);
        }
        let output = self.output(requested_output, size)?;
        self.emit(Opcode::Load, Some(output), &[Self::space_id(space), ptr]);
        Ok(output)
    }

    fn lower_builtin(
        &mut self,
        expr: &Expression,
        builtin: Builtin,
        args: &[Expression],
        requested_output: Option<Varnode>,
    ) -> Result<Varnode, PcodeLowerError> {
        let opcode = match builtin {
            Builtin::Carry => Opcode::IntCarry,
            Builtin::Scarry => Opcode::IntSCarry,
            Builtin::Sborrow => Opcode::IntSBorrow,
            Builtin::Nan => Opcode::FloatNan,
            Builtin::Abs => Opcode::FloatAbs,
            Builtin::Sqrt => Opcode::FloatSqrt,
            Builtin::Floor => Opcode::FloatFloor,
            Builtin::Ceil => Opcode::FloatCeil,
            Builtin::Round => Opcode::FloatRound,
            Builtin::Int2Float => Opcode::FloatInt2Float,
            Builtin::Float2Float => Opcode::FloatFloat2Float,
            Builtin::Trunc => Opcode::FloatTrunc,
            Builtin::Zext => Opcode::IntZext,
            Builtin::Sext => Opcode::IntSext,
            Builtin::Popcount => Opcode::PopCount,
            Builtin::Lzcount => Opcode::LzCount,
            Builtin::Cpool => Opcode::CpoolRef,
            Builtin::NewObject => Opcode::New,
        };
        let size = requested_output
            .map(|output| output.size)
            .or(expr.size)
            .or(match builtin {
                Builtin::Carry | Builtin::Scarry | Builtin::Sborrow | Builtin::Nan => Some(1),
                _ => None,
            })
            .ok_or(PcodeLowerError::UnknownSize)?;
        let output = self.output(requested_output, size)?;
        // The carry-family builtins return a boolean but consume equally-sized
        // integer operands. Their result width therefore cannot provide the
        // context required by a nested `zext`; carry the first operand's width
        // into the remaining operands instead.
        let inputs = if matches!(builtin, Builtin::Carry | Builtin::Scarry | Builtin::Sborrow)
            && !args.is_empty()
        {
            // The first operand can be an unsized literal (`sborrow(0, RAX)`
            // in x86 `NEG`). Carry-family operands must all have the same
            // width, so derive it from any concrete operand before lowering.
            let operand_size = args
                .iter()
                .find_map(|arg| self.expr_size(arg))
                .ok_or(PcodeLowerError::UnknownSize)?;
            args.iter()
                .map(|arg| self.lower_expr_with_size(arg, operand_size))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            args.iter()
                .map(|arg| self.lower_expr(arg, None))
                .collect::<Result<Vec<_>, _>>()?
        };
        self.emit(opcode, Some(output), &inputs);
        Ok(output)
    }

    fn lower_unop(
        &mut self,
        expr: &Expression,
        op: UnaryOperator,
        operand: &Expression,
        requested_output: Option<Varnode>,
    ) -> Result<Varnode, PcodeLowerError> {
        if let UnaryOperator::AddressOf(size) = op {
            // An address symbol such as `inst_next` already *is* its address;
            // taking its address only fixes the width.
            if let ExpressionTy::SizedInt {
                value,
                size: literal_size,
            } = &operand.ty
            {
                let size = size
                    .or(*literal_size)
                    .or(operand.size)
                    .ok_or(PcodeLowerError::UnknownSize)?;
                return self.copy_if_requested(Varnode::constant(*value, size), requested_output);
            }
            let storage = self.storage_from_expr(operand)?;
            let size = size
                .or_else(|| self.context.address_size(storage.space))
                .ok_or(PcodeLowerError::UnknownSize)?;
            return self
                .copy_if_requested(Varnode::constant(storage.offset, size), requested_output);
        }
        let opcode = match op {
            UnaryOperator::LogicalNot => Opcode::BoolNegate,
            UnaryOperator::BitwiseNot => Opcode::IntNegate,
            UnaryOperator::Minus => Opcode::Int2Comp,
            UnaryOperator::FloatMinus => Opcode::FloatNeg,
            UnaryOperator::AddressOf(_) => unreachable!(),
        };
        // Unsized integer literals are polymorphic. Resolve the unary result
        // width before lowering its operand so `~8` can inherit the width of
        // its assignment (for example x86 `CLTS`), rather than failing while
        // lowering the literal without a consumer.
        let size = requested_output
            .map(|output| output.size)
            .or(expr.size)
            .or_else(|| (op == UnaryOperator::LogicalNot).then_some(1))
            .or_else(|| self.expr_size(operand))
            .ok_or(PcodeLowerError::UnknownSize)?;
        // Preserve a concrete operand's native width (notably BOOL_NEGATE,
        // whose input need not be one byte); only force the result width into
        // a width-less operand such as an integer literal.
        let input = if self.expr_size(operand).is_some() {
            self.lower_expr(operand, None)?
        } else {
            self.lower_expr_with_size(operand, size)?
        };
        let output = self.output(requested_output, size)?;
        self.emit(opcode, Some(output), &[input]);
        Ok(output)
    }

    fn lower_binop(
        &mut self,
        expr: &Expression,
        op: BinaryOperator,
        lhs: &Expression,
        rhs: &Expression,
        requested_output: Option<Varnode>,
    ) -> Result<Varnode, PcodeLowerError> {
        let (opcode, reverse) = binary_opcode(op);
        // An arithmetic result has the same width as its operands. Comparisons
        // and boolean operations instead produce one byte, so obtain their
        // operand width from either side. This supplies the context needed by
        // unsized SLEIGH literals and compound expressions (for example the
        // `2 * zext(DF)` in x86 MOVS pointer updates).
        let is_boolean = op.is_comparison()
            || matches!(
                op,
                BinaryOperator::LogicalXor | BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr
            );
        let input_size = if is_boolean {
            self.expr_size(lhs).or_else(|| self.expr_size(rhs))
        } else {
            requested_output
                .map(|output| output.size)
                .or(expr.size)
                .or_else(|| self.expr_size(lhs))
                .or_else(|| self.expr_size(rhs))
        };
        let mut inputs = match input_size {
            Some(size) => vec![
                self.lower_expr_with_size(lhs, size)?,
                self.lower_expr_with_size(rhs, size)?,
            ],
            None => vec![self.lower_expr(lhs, None)?, self.lower_expr(rhs, None)?],
        };
        if reverse {
            inputs.swap(0, 1);
        }
        let size = requested_output
            .map(|output| output.size)
            .or(expr.size)
            .or_else(|| op.is_comparison().then_some(1))
            .or(input_size)
            .ok_or(PcodeLowerError::UnknownSize)?;
        let output = self.output(requested_output, size)?;
        if (op.is_comparison()
            || matches!(
                op,
                BinaryOperator::LogicalXor | BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr
            ))
            && output.size != 1
        {
            return Err(PcodeLowerError::InvalidBooleanSize(output.size));
        }
        if inputs[0].size != inputs[1].size {
            return Err(PcodeLowerError::InputSizeMismatch {
                operation: "binary operation",
                left: inputs[0].size,
                right: inputs[1].size,
            });
        }
        if !op.is_comparison()
            && !matches!(
                op,
                BinaryOperator::LogicalXor | BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr
            )
            && output.size != inputs[0].size
        {
            return Err(PcodeLowerError::CopySizeMismatch {
                input: inputs[0].size,
                output: output.size,
            });
        }
        self.emit(opcode, Some(output), &inputs);
        Ok(output)
    }

    /// Lowers an operand in a context that requires `size` bytes.
    ///
    /// SLEIGH permits a narrow register or temporary as a shift count or bit
    /// index for a wider value. Raw p-code does not: both inputs of these
    /// operations must have the same width. Requesting an output of `size`
    /// propagates that context into compound expressions, while
    /// [`copy_if_requested`](Self::copy_if_requested) inserts an explicit
    /// zero-extension or low-byte `SUBPIECE` for a directly stored value.
    fn lower_expr_with_size(
        &mut self,
        expr: &Expression,
        size: usize,
    ) -> Result<Varnode, PcodeLowerError> {
        // SLEIGH integer literals are polymorphic in raw p-code: the
        // surrounding operation determines their varnode width (for example
        // `RAX + 1`). This applies even when parsing retained a literal's
        // minimal source width.
        if let ExpressionTy::SizedInt { value, .. } = &expr.ty {
            return Ok(Varnode::constant(*value, size));
        }
        if self.expr_size(expr) == Some(size) {
            return self.lower_expr(expr, None);
        }
        let output = self.allocate_unique(size)?;
        self.lower_expr(expr, Some(output))
    }

    fn lower_range(
        &mut self,
        range: &Range,
        requested_output: Option<Varnode>,
    ) -> Result<Varnode, PcodeLowerError> {
        let input = self.lower_expr(&range.value, None)?;
        let (start, bits) = range_params(range)?;
        self.extract_range(input, start, bits, requested_output)
    }

    fn lower_named_bitrange(
        &mut self,
        id: BitRangeFieldId,
        requested_output: Option<Varnode>,
    ) -> Result<Varnode, PcodeLowerError> {
        let info = self
            .context
            .bitrange_info(id)
            .ok_or(PcodeLowerError::Unsupported("an unknown named bit range"))?;
        self.extract_range(info.storage, info.start, info.size, requested_output)
    }

    fn extract_range(
        &mut self,
        input: Varnode,
        start: usize,
        bits: usize,
        requested_output: Option<Varnode>,
    ) -> Result<Varnode, PcodeLowerError> {
        let result_size = Self::validate_range(input, start, bits)?;
        if let Some(output) = requested_output {
            if output.size != result_size {
                return Err(PcodeLowerError::CopySizeMismatch {
                    input: result_size,
                    output: output.size,
                });
            }
        }
        let shifted = self.allocate_unique(input.size)?;
        self.emit(
            Opcode::IntRight,
            Some(shifted),
            &[input, Varnode::constant(start as u64, input.size)],
        );
        let masked = self.allocate_unique(input.size)?;
        self.emit(
            Opcode::IntAnd,
            Some(masked),
            &[shifted, Varnode::constant(Self::mask(bits)?, input.size)],
        );
        let output = self.output(requested_output, result_size)?;
        self.emit(
            Opcode::SubPiece,
            Some(output),
            &[masked, Varnode::constant(0, 8)],
        );
        Ok(output)
    }

    fn lower_range_assignment(
        &mut self,
        range: &Range,
        rhs: &Expression,
    ) -> Result<(), PcodeLowerError> {
        if let ExpressionTy::Load(load) = &range.value.ty {
            return self.lower_load_range_assignment(load, range, rhs);
        }
        let storage = self.storage_from_expr(&range.value)?;
        let (start, bits) = range_params(range)?;
        self.insert_range(storage, start, bits, rhs)
    }

    /// Lowers a bit-range write into a memory load as load/modify/store. SLEIGH
    /// uses this form for packed MMX lanes backed by private RAM, where an
    /// address-of expression cannot name a raw-p-code varnode directly.
    fn lower_load_range_assignment(
        &mut self,
        load: &Load,
        range: &Range,
        rhs: &Expression,
    ) -> Result<(), PcodeLowerError> {
        let storage = self.lower_load(&range.value, load, None)?;
        let (start, bits) = range_params(range)?;
        Self::validate_range(storage, start, bits)?;
        if storage.size > 8 {
            return Err(PcodeLowerError::InvalidRange {
                start,
                size: bits,
                storage_bits: storage.size.saturating_mul(8),
            });
        }
        let value = self.lower_expr_with_size(rhs, bits.div_ceil(8))?;
        if value.size > storage.size {
            return Err(PcodeLowerError::InputSizeMismatch {
                operation: "bit-range assignment",
                left: storage.size,
                right: value.size,
            });
        }
        let extended = if value.size == storage.size {
            value
        } else {
            let output = self.allocate_unique(storage.size)?;
            self.emit(Opcode::IntZext, Some(output), &[value]);
            output
        };
        let inserted = self.allocate_unique(storage.size)?;
        self.emit(
            Opcode::IntAnd,
            Some(inserted),
            &[extended, Varnode::constant(Self::mask(bits)?, storage.size)],
        );
        let shifted = self.allocate_unique(storage.size)?;
        self.emit(
            Opcode::IntLeft,
            Some(shifted),
            &[inserted, Varnode::constant(start as u64, storage.size)],
        );
        let clear_mask = !(Self::mask(bits)? << start);
        let kept = self.allocate_unique(storage.size)?;
        self.emit(
            Opcode::IntAnd,
            Some(kept),
            &[storage, Varnode::constant(clear_mask, storage.size)],
        );
        let result = self.allocate_unique(storage.size)?;
        self.emit(Opcode::IntOr, Some(result), &[kept, shifted]);

        let space = self.load_space(load)?;
        if space == SPACE_CONST {
            return Err(PcodeLowerError::Unsupported("a store to constant space"));
        }
        let ptr = self.lower_expr(&load.ptr, None)?;
        self.validate_pointer(space, ptr)?;
        self.emit(Opcode::Store, None, &[Self::space_id(space), ptr, result]);
        Ok(())
    }

    fn insert_range(
        &mut self,
        storage: Varnode,
        start: usize,
        bits: usize,
        rhs: &Expression,
    ) -> Result<(), PcodeLowerError> {
        Self::validate_range(storage, start, bits)?;
        // Inserting needs a full-width clear mask. Constants in this AST are
        // u64, so zero-extending one into larger storage would incorrectly
        // clear every high bit.
        if storage.size > 8 {
            return Err(PcodeLowerError::InvalidRange {
                start,
                size: bits,
                storage_bits: storage.size.saturating_mul(8),
            });
        }
        // A range assignment fixes the RHS width even when the RHS is a
        // user-op result whose source expression does not carry one.
        let value = self.lower_expr_with_size(rhs, bits.div_ceil(8))?;
        if value.size > storage.size {
            return Err(PcodeLowerError::InputSizeMismatch {
                operation: "bit-range assignment",
                left: storage.size,
                right: value.size,
            });
        }
        let extended = if value.size == storage.size {
            value
        } else {
            let output = self.allocate_unique(storage.size)?;
            self.emit(Opcode::IntZext, Some(output), &[value]);
            output
        };
        let inserted = self.allocate_unique(storage.size)?;
        self.emit(
            Opcode::IntAnd,
            Some(inserted),
            &[extended, Varnode::constant(Self::mask(bits)?, storage.size)],
        );
        let shifted = self.allocate_unique(storage.size)?;
        self.emit(
            Opcode::IntLeft,
            Some(shifted),
            &[inserted, Varnode::constant(start as u64, storage.size)],
        );
        let clear_mask = !(Self::mask(bits)? << start);
        let kept = self.allocate_unique(storage.size)?;
        self.emit(
            Opcode::IntAnd,
            Some(kept),
            &[storage, Varnode::constant(clear_mask, storage.size)],
        );
        self.emit(Opcode::IntOr, Some(storage), &[kept, shifted]);
        Ok(())
    }

    fn validate_range(
        storage: Varnode,
        start: usize,
        size: usize,
    ) -> Result<usize, PcodeLowerError> {
        let storage_bits = storage
            .size
            .checked_mul(8)
            .ok_or(PcodeLowerError::InvalidRange {
                start,
                size,
                storage_bits: usize::MAX,
            })?;
        if size == 0 || size > 64 || start.checked_add(size).is_none_or(|end| end > storage_bits) {
            return Err(PcodeLowerError::InvalidRange {
                start,
                size,
                storage_bits,
            });
        }
        Ok(size.div_ceil(8))
    }

    fn mask(bits: usize) -> Result<u64, PcodeLowerError> {
        match bits {
            1..=63 => Ok((1u64 << bits) - 1),
            64 => Ok(u64::MAX),
            _ => Err(PcodeLowerError::InvalidRange {
                start: 0,
                size: bits,
                storage_bits: 64,
            }),
        }
    }

    fn validate_pointer(&self, space: SpaceId, ptr: Varnode) -> Result<(), PcodeLowerError> {
        let expected = self
            .context
            .address_size(space)
            .ok_or(PcodeLowerError::UnresolvedSpace)?;
        Self::checked_size(expected)?;
        if ptr.size != expected {
            return Err(PcodeLowerError::AddressSizeMismatch {
                expected,
                actual: ptr.size,
            });
        }
        Ok(())
    }

    fn storage_from_expr(&mut self, expr: &Expression) -> Result<Varnode, PcodeLowerError> {
        match &expr.ty {
            ExpressionTy::Ident(ident) => self.storage_for_ident(ident.clone(), expr.size),
            _ => Err(PcodeLowerError::Unsupported(
                "address-of a non-varnode expression",
            )),
        }
    }

    fn storage_for_ident(
        &mut self,
        ident: Ident,
        size: Option<usize>,
    ) -> Result<Varnode, PcodeLowerError> {
        match ident {
            Ident::Register(id) => self
                .context
                .register_varnode(id)
                .ok_or(PcodeLowerError::UnknownRegister(id)),
            Ident::Named(id) => {
                let size = self.plan.local_sizes.get(&id).copied().or(size);
                if let Some(varnode) = self.locals.get(&id) {
                    if let Some(size) = size
                        && size != varnode.size
                    {
                        return Err(PcodeLowerError::CopySizeMismatch {
                            input: varnode.size,
                            output: size,
                        });
                    }
                    return Ok(*varnode);
                }
                let varnode = self.allocate_unique(size.ok_or(PcodeLowerError::UnknownSize)?)?;
                self.locals.insert(id, varnode);
                Ok(varnode)
            }
            Ident::BitRange(_) => Err(PcodeLowerError::Unsupported("a named bit range")),
            Ident::Field(_) => Err(PcodeLowerError::UnresolvedIdentifier("field")),
            Ident::Table(_) => Err(PcodeLowerError::UnresolvedIdentifier("table")),
            Ident::Global(_) => Err(PcodeLowerError::UnresolvedIdentifier("global")),
        }
    }

    fn lower_userop_inputs(
        &mut self,
        id: PCodeOpId,
        args: &[Expression],
    ) -> Result<Vec<Varnode>, PcodeLowerError> {
        let mut inputs = Vec::with_capacity(args.len() + 1);
        inputs.push(Varnode::constant(usize::from(id) as u64, 4));
        inputs.extend(
            args.iter()
                .map(|arg| self.lower_expr(arg, None))
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(inputs)
    }

    fn copy_if_requested(
        &mut self,
        input: Varnode,
        requested_output: Option<Varnode>,
    ) -> Result<Varnode, PcodeLowerError> {
        match requested_output {
            Some(output) if output != input && input.size == output.size => {
                self.emit(Opcode::Copy, Some(output), &[input]);
                Ok(output)
            }
            Some(output) if input.size < output.size => {
                self.emit(Opcode::IntZext, Some(output), &[input]);
                Ok(output)
            }
            Some(output) if input.size > output.size => {
                self.emit(
                    Opcode::SubPiece,
                    Some(output),
                    &[input, Varnode::constant(0, 8)],
                );
                Ok(output)
            }
            Some(output) => Ok(output),
            None => Ok(input),
        }
    }

    fn output(
        &mut self,
        requested_output: Option<Varnode>,
        size: usize,
    ) -> Result<Varnode, PcodeLowerError> {
        match requested_output {
            Some(output) => {
                Self::checked_size(output.size)?;
                Ok(output)
            }
            None => self.allocate_unique(size),
        }
    }

    fn allocate_unique(&mut self, size: usize) -> Result<Varnode, PcodeLowerError> {
        Self::checked_size(size)?;
        let offset = self.next_unique;
        self.next_unique = self
            .next_unique
            .checked_add(size as u64)
            .ok_or(PcodeLowerError::UniqueSpaceOverflow)?;
        Ok(Varnode::new(self.context.unique_space(), offset, size))
    }

    fn label_id(&self, label: &str) -> Result<LabelId, PcodeLowerError> {
        self.plan
            .label_id(label)
            .ok_or_else(|| PcodeLowerError::UnknownLabel(Box::from(label)))
    }

    fn checked_size(size: usize) -> Result<(), PcodeLowerError> {
        if size == 0 {
            Err(PcodeLowerError::ZeroSize)
        } else {
            Ok(())
        }
    }

    fn space_id(space: SpaceId) -> Varnode {
        Varnode::constant(usize::from(space) as u64, 4)
    }
}

/// The read-only pass which produces a [`PcodePlan`].
///
/// It is generic over the statement span so a producer can run the same width
/// inference over its own *source* bodies, before any instruction is decoded,
/// rather than keeping a second implementation that can drift from this one.
struct Planner<'a, C: PcodeLoweringContext + ?Sized> {
    context: &'a C,
    plan: PcodePlan,
}

impl<'a, C: PcodeLoweringContext + ?Sized> Planner<'a, C> {
    fn plan(&mut self, ast: &PcodeAst) {
        self.plan.local_sizes = SizeInference::run(self.context, &ast.statements);
        self.plan_statements(ast);
    }

    /// Collects the facts that do not depend on local widths: the labels and
    /// the addresses this instruction reaches directly.
    fn plan_statements(&mut self, ast: &PcodeAst) {
        for statement in &ast.statements {
            match &statement.ty {
                AstNode::Label(label) => {
                    self.plan.declare_label(label);
                }
                AstNode::Branch { target } | AstNode::ConditionalBranch { target, .. } => {
                    // A target this pass cannot resolve is left out; emission
                    // reports it with the error it would have reported before.
                    if let LabelOrNode::Expr(expr) = target
                        && let Some(address) = self.direct_address(expr)
                    {
                        self.plan.declare_direct_branch(address);
                    }
                }
                AstNode::Call { target } => {
                    if let LabelOrNode::Expr(expr) = target
                        && let Some(address) = self.direct_address(expr)
                    {
                        self.plan.declare_direct_call(address);
                    }
                }
                _ => {}
            }
        }

        // Only labels may follow the last operation-producing statement, so
        // the trailing run of labels is exactly the terminal one.
        for statement in ast.statements.iter().rev() {
            let AstNode::Label(label) = &statement.ty else {
                break;
            };
            if let Some(id) = self.plan.label_id(label) {
                self.plan.terminal[id.index()] = true;
            }
        }
    }

    fn direct_address<S>(&self, target: &Expression<S>) -> Option<u64> {
        match target.ty {
            ExpressionTy::SizedInt { value, .. } => Some(value),
            _ => None,
        }
    }
}

/// The width-inference pass, shared by specification-compile time and by
/// per-instruction planning.
///
/// It is generic over the statement span so a producer can run it over its own
/// *source* bodies, and over the width domain so those bodies can be resolved
/// before the values a decode substitutes into them are known.
struct SizeInference<'a, C: PcodeLoweringContext + ?Sized, W: Width> {
    context: &'a C,
    sizes: HashMap<LocalVarId, W>,
}

impl<'a, C: PcodeLoweringContext + ?Sized, W: Width> SizeInference<'a, C, W> {
    fn run<S>(context: &'a C, statements: &[Ast<S>]) -> HashMap<LocalVarId, W> {
        let mut inference = Self {
            context,
            sizes: HashMap::new(),
        };
        inference.infer(statements);
        inference.sizes
    }

    /// Resolve local widths from their uses. A forward-only allocator cannot
    /// size, for example, `v = 255 & 31` until a later `word << v` reveals
    /// that `v` is a word-wide shift count.
    fn infer<S>(&mut self, statements: &[Ast<S>]) {
        // Each pass can discover at least one previously unknown local. The
        // extra pass propagates that discovery through a chain of locals.
        for _ in 0..=statements.len() {
            let before = self.sizes.len();
            for statement in statements {
                self.constrain_statement(&statement.ty);
            }
            if self.sizes.len() == before {
                break;
            }
        }
    }

    fn constrain_statement<S>(&mut self, statement: &AstNode<S>) {
        match statement {
            AstNode::Assignment { lhs, size, rhs } => {
                // Comparisons normally infer a one-byte result. A different
                // explicit expression size must still reach lowering so it is
                // rejected as an invalid raw boolean output.
                let comparison_size =
                    matches!(&rhs.ty, ExpressionTy::Binop(binop) if binop.op.is_comparison())
                        .then_some(rhs.size)
                        .flatten()
                        .filter(|&size| size != 1)
                        .map(W::fixed);
                let expected = (*size)
                    .map(W::fixed)
                    .or_else(|| self.storage_size(lhs))
                    .or(comparison_size);
                let inferred = self.constrain_expr(rhs, expected);
                if let Ident::Named(id) = lhs
                    && let Some(size) = expected.or(inferred)
                {
                    self.sizes.entry(*id).or_insert(size);
                }
            }
            AstNode::LoadAssignment { lhs, rhs, .. } => {
                let space = self.load_space(lhs).ok();
                if let Some(space) = space {
                    self.constrain_expr(&lhs.ptr, self.context.address_size(space).map(W::fixed));
                }
                self.constrain_expr(rhs, lhs.size.map(W::fixed));
            }
            AstNode::RangeAssignment { lhs, rhs, .. } => {
                if let Ok((_, bits)) = range_params(lhs) {
                    self.constrain_expr(rhs, Some(W::fixed(bits.div_ceil(8))));
                }
            }
            AstNode::ConditionalBranch { condition, .. } => {
                self.constrain_expr(condition, Some(W::fixed(1)));
            }
            AstNode::BranchIndirect { target }
            | AstNode::CallIndirect { target }
            | AstNode::Return { target } => {
                self.constrain_expr(
                    target,
                    self.context
                        .address_size(self.context.default_space())
                        .map(W::fixed),
                );
            }
            AstNode::Expression(expr) => {
                self.constrain_expr(expr, None);
            }
            AstNode::Build(_)
            | AstNode::DelaySlot(_)
            | AstNode::DeferredBuild(_)
            | AstNode::Label(_)
            | AstNode::Branch { .. }
            | AstNode::Call { .. }
            | AstNode::Export(_) => {}
        }
    }

    /// Applies an optional consumer width to `expr` and returns any concrete
    /// output width known after that constraint. Integer literals intentionally
    /// do not establish a width on their own.
    fn constrain_expr<S>(&mut self, expr: &Expression<S>, expected: Option<W>) -> Option<W> {
        match &expr.ty {
            ExpressionTy::SizedInt { .. } => expected,
            ExpressionTy::Ident(Ident::Named(id)) => {
                if let Some(&size) = self.sizes.get(id) {
                    Some(size)
                } else if let Some(size) = expected {
                    self.sizes.insert(*id, size);
                    Some(size)
                } else {
                    None
                }
            }
            ExpressionTy::Ident(ident) => self.storage_size(ident),
            ExpressionTy::Load(load) => {
                if let Ok(space) = self.load_space(load) {
                    self.constrain_expr(&load.ptr, self.context.address_size(space).map(W::fixed));
                }
                load.size.map(W::fixed).or(expected)
            }
            ExpressionTy::SubPieceLsb { src, count } => {
                self.constrain_expr(src, None);
                Some(W::fixed(*count))
            }
            ExpressionTy::SubPieceMsb { src, count } => {
                // Truncation is arithmetic on a width, so a still-symbolic
                // operand width yields no constraint rather than a wrong one.
                let size = expected
                    .or_else(|| Some(W::fixed(self.expr_size(src)?.size()?.checked_sub(*count)?)));
                let source = size
                    .and_then(|size| size.size())
                    .map(|size| W::fixed(size + count));
                self.constrain_expr(src, source);
                size
            }
            ExpressionTy::Range(range) => {
                let size = match range.size {
                    RangeParam::Literal(bits) => Some(W::fixed(bits.div_ceil(8))),
                    RangeParam::MacroArg(_) => expected,
                };
                self.constrain_expr(&range.value, None);
                size
            }
            ExpressionTy::FunctionCall { builtin, args } => {
                let boolean = matches!(
                    builtin,
                    Builtin::Carry | Builtin::Scarry | Builtin::Sborrow | Builtin::Nan
                );
                let size = boolean.then(|| W::fixed(1)).or(expected);
                let input_size = args.iter().find_map(|arg| self.constrain_expr(arg, None));
                if let Some(input_size) = input_size {
                    for arg in args {
                        self.constrain_expr(arg, Some(input_size));
                    }
                }
                size
            }
            ExpressionTy::PcodeOp { args, .. } => {
                for arg in args {
                    self.constrain_expr(arg, None);
                }
                expected
            }
            ExpressionTy::Unop(unop) => match unop.op {
                UnaryOperator::LogicalNot => {
                    let size = self.constrain_expr(&unop.e, None);
                    self.constrain_expr(&unop.e, size);
                    Some(W::fixed(1))
                }
                UnaryOperator::AddressOf(size) => size.map(W::fixed).or_else(|| {
                    self.storage_from_expr_size(&unop.e)
                        .and_then(|storage| self.context.address_size(storage.space))
                        .map(W::fixed)
                }),
                _ => {
                    let size = expected.or_else(|| self.constrain_expr(&unop.e, None));
                    self.constrain_expr(&unop.e, size);
                    size
                }
            },
            ExpressionTy::Binop(binop) => {
                let boolean = binop.op.is_comparison()
                    || matches!(
                        binop.op,
                        BinaryOperator::LogicalXor
                            | BinaryOperator::LogicalAnd
                            | BinaryOperator::LogicalOr
                    );
                let input_size = self
                    .constrain_expr(&binop.lhs, None)
                    .or_else(|| self.constrain_expr(&binop.rhs, None));
                let input_size = if boolean {
                    input_size
                } else {
                    expected.or(input_size)
                };
                self.constrain_expr(&binop.lhs, input_size);
                self.constrain_expr(&binop.rhs, input_size);
                if boolean {
                    Some(W::fixed(1))
                } else {
                    input_size
                }
            }
            ExpressionTy::MacroCall { .. } | ExpressionTy::DeferredCall { .. } => expected,
        }
    }
}

impl<'a, C: PcodeLoweringContext + ?Sized, W: Width> Sizing<W> for SizeInference<'a, C, W> {
    type Ctx = C;

    fn context(&self) -> &C {
        self.context
    }

    fn local_size(&self, id: &LocalVarId) -> Option<W> {
        self.sizes.get(id).copied()
    }
}

impl<C: PcodeLoweringContext + ?Sized, S: PcodeSink + ?Sized> Sizing<usize>
    for Lowerer<'_, '_, '_, C, S>
{
    type Ctx = C;

    fn context(&self) -> &C {
        self.context
    }

    fn local_size(&self, id: &LocalVarId) -> Option<usize> {
        self.plan
            .local_sizes
            .get(id)
            .copied()
            .or_else(|| self.locals.get(id).map(|varnode| varnode.size))
    }
}

/// Width and space queries shared by planning and emission. Both phases must
/// answer them identically, so they have one implementation parameterized by
/// how each phase knows a local's width.
trait Sizing<W: Width> {
    type Ctx: PcodeLoweringContext + ?Sized;

    fn context(&self) -> &Self::Ctx;

    /// The width of a local variable, if it is known in this phase.
    fn local_size(&self, id: &LocalVarId) -> Option<W>;

    fn expr_size<S>(&self, expr: &Expression<S>) -> Option<W> {
        expr.size.map(W::fixed).or(match &expr.ty {
            ExpressionTy::SizedInt { size, .. } => size.map(W::fixed),
            ExpressionTy::Ident(ident) => self.storage_size(ident),
            ExpressionTy::Load(load) => load.size.map(W::fixed),
            ExpressionTy::SubPieceLsb { count, .. } => Some(W::fixed(*count)),
            ExpressionTy::SubPieceMsb { src, count } => {
                Some(W::fixed(self.expr_size(src)?.size()?.checked_sub(*count)?))
            }
            ExpressionTy::Range(Range {
                size: RangeParam::Literal(bits),
                ..
            }) => Some(W::fixed(bits.div_ceil(8))),
            ExpressionTy::Range(Range {
                size: RangeParam::MacroArg(_),
                ..
            }) => None,
            ExpressionTy::FunctionCall {
                builtin: Builtin::Carry | Builtin::Scarry | Builtin::Sborrow | Builtin::Nan,
                ..
            } => Some(W::fixed(1)),
            ExpressionTy::FunctionCall { .. } => None,
            ExpressionTy::Unop(unop) if unop.op == UnaryOperator::LogicalNot => Some(W::fixed(1)),
            ExpressionTy::Unop(unop) => self.expr_size(&unop.e),
            ExpressionTy::Binop(binop) if binop.op.is_comparison() => Some(W::fixed(1)),
            ExpressionTy::Binop(binop) => self
                .expr_size(&binop.lhs)
                .or_else(|| self.expr_size(&binop.rhs)),
            ExpressionTy::PcodeOp { .. }
            | ExpressionTy::MacroCall { .. }
            | ExpressionTy::DeferredCall { .. } => None,
        })
    }

    fn storage_size(&self, ident: &Ident) -> Option<W> {
        match ident {
            Ident::Register(id) => self
                .context()
                .register_varnode(*id)
                .map(|varnode| W::fixed(varnode.size)),
            Ident::BitRange(id) => self
                .context()
                .bitrange_info(*id)
                .map(|info| W::fixed(info.size.div_ceil(8))),
            Ident::Named(id) => self.local_size(id),
            // An operand's width is only known once a decode substitutes its
            // value. A symbolic domain names it instead of losing it.
            Ident::Table(id) => W::operand(OperandKey::Table(*id)),
            Ident::Field(id) => W::operand(OperandKey::Field(*id)),
            Ident::Global(_) => None,
        }
    }

    fn storage_from_expr_size<S>(&self, expr: &Expression<S>) -> Option<Varnode> {
        match &expr.ty {
            ExpressionTy::Ident(Ident::Register(id)) => self.context().register_varnode(*id),
            ExpressionTy::Ident(Ident::BitRange(id)) => {
                self.context().bitrange_info(*id).map(|info| info.storage)
            }
            _ => None,
        }
    }

    fn load_space<S>(&self, load: &Load<S>) -> Result<SpaceId, PcodeLowerError> {
        match &load.space {
            None => Ok(self.context().default_space()),
            Some(crate::PcodeSpaceRef::Resolved(space)) => Ok(*space),
            Some(crate::PcodeSpaceRef::Deferred(_)) => Err(PcodeLowerError::UnresolvedSpace),
        }
    }
}

/// Reads a bit range's literal start and width.
///
/// A macro-argument range must have been substituted during expansion.
fn range_params<S>(range: &Range<S>) -> Result<(usize, usize), PcodeLowerError> {
    let RangeParam::Literal(start) = range.start else {
        return Err(PcodeLowerError::UnresolvedRangeParameter);
    };
    let RangeParam::Literal(size) = range.size else {
        return Err(PcodeLowerError::UnresolvedRangeParameter);
    };
    Ok((start, size))
}

fn binary_opcode(op: BinaryOperator) -> (Opcode, bool) {
    use BinaryOperator::*;
    match op {
        Mul => (Opcode::IntMult, false),
        Div => (Opcode::IntDiv, false),
        SignedDiv => (Opcode::IntSDiv, false),
        Mod => (Opcode::IntRem, false),
        SignedMod => (Opcode::IntSRem, false),
        FloatDiv => (Opcode::FloatDiv, false),
        FloatMul => (Opcode::FloatMult, false),
        Add => (Opcode::IntAdd, false),
        Sub => (Opcode::IntSub, false),
        FloatAdd => (Opcode::FloatAdd, false),
        FloatSub => (Opcode::FloatSub, false),
        LeftShift => (Opcode::IntLeft, false),
        RightShift => (Opcode::IntRight, false),
        SignedRightShift => (Opcode::IntSRight, false),
        SignedLessThan => (Opcode::IntSLess, false),
        SignedGreaterThan => (Opcode::IntSLess, true),
        SignedLessEqual => (Opcode::IntSLessEqual, false),
        SignedGreaterEqual => (Opcode::IntSLessEqual, true),
        LessEqual => (Opcode::IntLessEqual, false),
        GreaterEqual => (Opcode::IntLessEqual, true),
        LessThan => (Opcode::IntLess, false),
        GreaterThan => (Opcode::IntLess, true),
        FloatLessEqual => (Opcode::FloatLessEqual, false),
        FloatGreaterEqual => (Opcode::FloatLessEqual, true),
        FloatLessThan => (Opcode::FloatLess, false),
        FloatGreaterThan => (Opcode::FloatLess, true),
        Equal => (Opcode::IntEqual, false),
        NotEqual => (Opcode::IntNotEqual, false),
        FloatEqual => (Opcode::FloatEqual, false),
        FloatNotEqual => (Opcode::FloatNotEqual, false),
        LogicalXor => (Opcode::BoolXor, false),
        LogicalAnd => (Opcode::BoolAnd, false),
        LogicalOr => (Opcode::BoolOr, false),
        BitwiseXor => (Opcode::IntXor, false),
        BitwiseOr => (Opcode::IntOr, false),
        BitwiseAnd => (Opcode::IntAnd, false),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BitRangeInfo, InstructionPcode, LabelId, LocalSizes, Opcode, PcodeLowerError,
        PcodeLoweringContext, PcodeOp, PcodeSink, Varnode, emit_instruction, lower_instruction,
        plan_instruction,
    };
    use crate::{
        Ast, AstNode, BinaryOperator, Binop, Expression, ExpressionTy, Ident, LabelOrNode, Load,
        LocalVarId, PCodeOpId, PcodeAst, PcodeSpaceRef, Range, RangeParam, RegisterId, SPACE_CONST,
        SpaceId,
    };
    use std::collections::HashMap;

    struct Context;

    impl PcodeLoweringContext for Context {
        fn default_space(&self) -> SpaceId {
            SpaceId::new(1)
        }

        fn unique_space(&self) -> SpaceId {
            SpaceId::new(2)
        }

        fn register_varnode(&self, id: RegisterId) -> Option<Varnode> {
            Some(Varnode::new(SpaceId::new(3), usize::from(id) as u64 * 4, 4))
        }

        fn bitrange_info(&self, _id: crate::BitRangeFieldId) -> Option<BitRangeInfo> {
            None
        }

        fn address_size(&self, space: SpaceId) -> Option<usize> {
            Some(if space == SpaceId::new(4) { 4 } else { 8 })
        }
    }

    fn int(value: u64, size: usize) -> Expression {
        Expression {
            ty: ExpressionTy::SizedInt {
                value,
                size: Some(size),
            },
            size: Some(size),
            span: (),
        }
    }

    fn ident(id: RegisterId) -> Expression {
        Expression {
            ty: ExpressionTy::Ident(Ident::Register(id)),
            size: Some(4),
            span: (),
        }
    }

    fn ast(nodes: Vec<AstNode>) -> PcodeAst {
        PcodeAst {
            statements: nodes.into_iter().map(Ast::from).collect(),
        }
    }

    #[test]
    fn varnodes_distinguish_constants_from_storage() {
        let constant = Varnode::constant(0x1234, 4);
        let storage = Varnode::new(SpaceId::new(2), 0x1234, 4);
        assert_eq!(constant.space, SPACE_CONST);
        assert!(constant.is_constant());
        assert!(!storage.is_constant());
    }

    #[test]
    fn opcode_inventory_identifies_analysis_only_operations() {
        assert_eq!(Opcode::ALL.len(), 72);
        assert_eq!(Opcode::Copy.ghidra_id(), 1);
        assert_eq!(Opcode::FloatLessEqual.ghidra_id(), 44);
        assert_eq!(Opcode::FloatNan.ghidra_id(), 46);
        assert_eq!(Opcode::LzCount.ghidra_id(), 73);
        assert!(Opcode::ALL.contains(&Opcode::Load));
        assert!(Opcode::ALL.contains(&Opcode::LzCount));
        assert!(Opcode::Load.is_raw_instruction_op());
        for opcode in [
            Opcode::MultiEqual,
            Opcode::Indirect,
            Opcode::Cast,
            Opcode::PtrAdd,
            Opcode::PtrSub,
            Opcode::SegmentOp,
            Opcode::Insert,
            Opcode::Extract,
        ] {
            assert!(!opcode.is_raw_instruction_op());
        }
    }

    #[test]
    fn flat_operations_preserve_input_order_and_round_trip() {
        let output = Varnode::new(SpaceId::new(1), 0, 4);
        let instruction = InstructionPcode {
            ops: vec![PcodeOp::new(
                Opcode::IntAdd,
                Some(output),
                vec![output, Varnode::constant(1, 4)],
            )],
        };
        assert!(!instruction.is_empty());
        let bytes =
            bincode::serde::encode_to_vec(&instruction, bincode::config::standard()).unwrap();
        let (decoded, _): (InstructionPcode, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(decoded, instruction);
        assert!(InstructionPcode::new().is_empty());
    }

    #[test]
    fn lower_assignment_emits_direct_arithmetic_output() {
        let rhs = Expression {
            ty: ExpressionTy::Binop(Binop {
                op: BinaryOperator::Add,
                lhs: Box::new(ident(RegisterId::new(1))),
                rhs: Box::new(int(1, 4)),
            }),
            size: Some(4),
            span: (),
        };
        let output = Varnode::new(SpaceId::new(3), 0, 4);
        let input = Varnode::new(SpaceId::new(3), 4, 4);
        let pcode = lower_instruction(
            &ast(vec![AstNode::Assignment {
                lhs: Ident::Register(RegisterId::new(0)),
                size: None,
                rhs,
            }]),
            &Context,
        )
        .unwrap();
        assert_eq!(
            pcode.ops,
            vec![PcodeOp::new(
                Opcode::IntAdd,
                Some(output),
                vec![input, Varnode::constant(1, 4)],
            )]
        );
    }

    #[test]
    fn lower_load_store_and_userop_use_ghidra_operand_order() {
        let ram = SpaceId::new(4);
        let pointer = ident(RegisterId::new(1));
        let load = Load {
            space: Some(PcodeSpaceRef::Resolved(ram)),
            size: Some(4),
            ptr: Box::new(pointer.clone()),
        };
        let pcode = lower_instruction(
            &ast(vec![
                AstNode::Assignment {
                    lhs: Ident::Register(RegisterId::new(0)),
                    size: None,
                    rhs: Expression {
                        ty: ExpressionTy::Load(load.clone()),
                        size: Some(4),
                        span: (),
                    },
                },
                AstNode::LoadAssignment {
                    lhs: load,
                    size: None,
                    rhs: int(9, 4),
                },
                AstNode::Expression(Expression {
                    ty: ExpressionTy::PcodeOp {
                        id: PCodeOpId::new(7),
                        args: vec![int(2, 4)],
                    },
                    size: None,
                    span: (),
                }),
            ]),
            &Context,
        )
        .unwrap();
        let r0 = Varnode::new(SpaceId::new(3), 0, 4);
        let r1 = Varnode::new(SpaceId::new(3), 4, 4);
        assert_eq!(
            pcode.ops,
            vec![
                PcodeOp::new(Opcode::Load, Some(r0), vec![Varnode::constant(4, 4), r1],),
                PcodeOp::new(
                    Opcode::Store,
                    None,
                    vec![Varnode::constant(4, 4), r1, Varnode::constant(9, 4)],
                ),
                PcodeOp::new(
                    Opcode::CallOther,
                    None,
                    vec![Varnode::constant(7, 4), Varnode::constant(2, 4)],
                ),
            ]
        );
    }

    /// Records the events of a streaming lift, keeping local branches symbolic.
    #[derive(Default)]
    struct Trace {
        events: Vec<String>,
    }

    impl PcodeSink for Trace {
        fn op(&mut self, opcode: Opcode, output: Option<Varnode>, inputs: &[Varnode]) {
            self.events
                .push(format!("{opcode:?} {output:?} {inputs:?}"));
        }

        fn label(&mut self, label: LabelId) {
            self.events.push(format!("label {}", label.index()));
        }

        fn branch_label(&mut self, opcode: Opcode, label: LabelId, condition: Option<Varnode>) {
            self.events.push(format!(
                "{opcode:?} -> label {} {condition:?}",
                label.index()
            ));
        }
    }

    fn branch_statements() -> Vec<AstNode> {
        vec![
            AstNode::ConditionalBranch {
                condition: ident(RegisterId::new(1)),
                target: LabelOrNode::Label("skip".into()),
            },
            AstNode::Call {
                target: LabelOrNode::Expr(int(0x2000, 8)),
            },
            AstNode::Branch {
                target: LabelOrNode::Expr(int(0x1000, 8)),
            },
            AstNode::Label("skip".into()),
            AstNode::Assignment {
                lhs: Ident::Register(RegisterId::new(1)),
                size: None,
                rhs: ident(RegisterId::new(2)),
            },
        ]
    }

    #[test]
    fn local_widths_can_be_inferred_from_a_body_before_planning() {
        let statements = vec![
            AstNode::Assignment {
                lhs: Ident::Named(LocalVarId(0)),
                size: None,
                rhs: ident(RegisterId::new(1)),
            },
            AstNode::Assignment {
                lhs: Ident::Register(RegisterId::new(2)),
                size: None,
                rhs: ExpressionTy::Ident(Ident::Named(LocalVarId(0))).with_size(4),
            },
        ];
        let ast = ast(statements.clone());

        // The same widths whether resolved from the body up front or by the
        // per-instruction planner.
        let sizes = super::infer_local_sizes(&ast.statements, &Context);
        assert_eq!(sizes.get(&LocalVarId(0)), Some(&4));

        let planned = super::plan_instruction_with(&ast, &Context, sizes).unwrap();
        let inferred = plan_instruction(&ast, &Context).unwrap();
        assert_eq!(planned.labels(), inferred.labels());

        // And supplied widths reach emission: the local becomes a 4-byte
        // unique, not an unsized-local error.
        let pcode = lower_instruction(&ast, &Context).unwrap();
        assert_eq!(
            pcode.ops[0].output,
            Some(Varnode::new(SpaceId::new(2), 0, 4))
        );
    }

    /// A width taken from a table operand must be *named*, not dropped: a
    /// pass that dropped it would size the local from the later statement and
    /// disagree with the per-instruction planner, which sees the substituted
    /// value first.
    #[test]
    fn symbolic_inference_names_an_operand_width_instead_of_losing_it() {
        let table = crate::TableId::new(7);
        let statements = ast(vec![
            AstNode::Assignment {
                lhs: Ident::Named(LocalVarId(0)),
                size: None,
                rhs: Expression {
                    ty: ExpressionTy::Ident(Ident::Table(table)),
                    size: None,
                    span: (),
                },
            },
            AstNode::Assignment {
                lhs: Ident::Register(RegisterId::new(1)),
                size: None,
                rhs: Expression {
                    ty: ExpressionTy::Binop(Binop {
                        op: BinaryOperator::Add,
                        lhs: Box::new(
                            ExpressionTy::Ident(Ident::Named(LocalVarId(0))).with_size(4),
                        ),
                        rhs: Box::new(ident(RegisterId::new(2))),
                    }),
                    size: None,
                    span: (),
                },
            },
        ])
        .statements;

        let symbolic: HashMap<LocalVarId, super::SymbolicWidth> =
            super::infer_local_sizes(&statements, &Context);
        assert_eq!(
            symbolic.get(&LocalVarId(0)),
            Some(&super::SymbolicWidth::SameAs(super::OperandKey::Table(
                table
            )))
        );

        // The concrete domain cannot name it, so it falls through to the
        // later use — which is exactly the disagreement the symbolic domain
        // exists to prevent.
        let concrete: LocalSizes = super::infer_local_sizes(&statements, &Context);
        assert_eq!(concrete.get(&LocalVarId(0)), Some(&4));
    }

    #[test]
    fn plan_reports_labels_and_out_of_instruction_targets() {
        let plan = plan_instruction(&ast(branch_statements()), &Context).unwrap();
        assert_eq!(plan.labels(), &[Box::<str>::from("skip")]);
        assert_eq!(plan.direct_branches(), &[0x1000]);
        assert_eq!(plan.direct_calls(), &[0x2000]);
    }

    #[test]
    fn streaming_emission_keeps_local_branch_targets_symbolic() {
        let ast = ast(branch_statements());
        let plan = plan_instruction(&ast, &Context).unwrap();
        let mut trace = Trace::default();
        emit_instruction(&ast, &Context, &plan, &mut trace).unwrap();

        assert_eq!(
            trace.events[0],
            "CBranch -> label 0 Some(Varnode { space: SpaceId(3), offset: 4, size: 4 })"
        );
        assert_eq!(trace.events[3], "label 0");
        assert_eq!(trace.events.len(), 5);

        // The collecting API resolves the same branch into a relative target.
        let pcode = lower_instruction(&ast, &Context).unwrap();
        assert_eq!(pcode.ops[0].opcode, Opcode::CBranch);
        assert_eq!(pcode.ops[0].inputs[0], Varnode::constant(3, 8));
    }

    #[test]
    fn plan_omits_unresolvable_direct_targets() {
        let plan = plan_instruction(
            &ast(vec![AstNode::Branch {
                target: LabelOrNode::Expr(ident(RegisterId::new(1))),
            }]),
            &Context,
        )
        .unwrap();
        assert!(plan.direct_branches().is_empty());
        assert_eq!(
            lower_instruction(
                &ast(vec![AstNode::Branch {
                    target: LabelOrNode::Expr(ident(RegisterId::new(1))),
                }]),
                &Context,
            )
            .unwrap_err(),
            PcodeLowerError::InvalidDirectTarget
        );
    }

    #[test]
    fn branching_to_an_undefined_label_is_rejected() {
        let error = lower_instruction(
            &ast(vec![AstNode::Branch {
                target: LabelOrNode::Label("missing".into()),
            }]),
            &Context,
        )
        .unwrap_err();
        assert_eq!(error, PcodeLowerError::UnknownLabel("missing".into()));
    }

    #[test]
    fn duplicate_labels_are_rejected() {
        let error = lower_instruction(
            &ast(vec![
                AstNode::Label("here".into()),
                AstNode::Label("here".into()),
            ]),
            &Context,
        )
        .unwrap_err();
        assert_eq!(error, PcodeLowerError::DuplicateLabel("here".into()));
    }

    #[test]
    fn lower_labels_are_pcode_relative_and_comparisons_reverse_greater_than() {
        let comparison = Expression {
            ty: ExpressionTy::Binop(Binop {
                op: BinaryOperator::GreaterThan,
                lhs: Box::new(ident(RegisterId::new(1))),
                rhs: Box::new(ident(RegisterId::new(2))),
            }),
            size: Some(1),
            span: (),
        };
        let pcode = lower_instruction(
            &ast(vec![
                AstNode::Label("loop".into()),
                AstNode::Assignment {
                    lhs: Ident::Named(LocalVarId(0)),
                    size: None,
                    rhs: comparison,
                },
                AstNode::Branch {
                    target: LabelOrNode::Label("loop".into()),
                },
            ]),
            &Context,
        )
        .unwrap();
        assert_eq!(pcode.ops.len(), 2);
        assert_eq!(pcode.ops[0].opcode, Opcode::IntLess);
        assert_eq!(
            pcode.ops[0].inputs,
            vec![
                Varnode::new(SpaceId::new(3), 8, 4),
                Varnode::new(SpaceId::new(3), 4, 4),
            ]
        );
        assert_eq!(
            pcode.ops[0].output,
            Some(Varnode::new(SpaceId::new(2), 0, 1))
        );
        assert_eq!(pcode.ops[1].opcode, Opcode::Branch);
        assert_eq!(pcode.ops[1].inputs, vec![Varnode::constant(u64::MAX, 8)]);
    }

    #[test]
    fn lower_named_bitranges_as_raw_read_modify_write() {
        struct BitRangeContext;
        impl PcodeLoweringContext for BitRangeContext {
            fn default_space(&self) -> SpaceId {
                SpaceId::new(1)
            }
            fn unique_space(&self) -> SpaceId {
                SpaceId::new(2)
            }
            fn register_varnode(&self, id: RegisterId) -> Option<Varnode> {
                Some(Varnode::new(SpaceId::new(3), usize::from(id) as u64 * 4, 4))
            }
            fn bitrange_info(&self, id: crate::BitRangeFieldId) -> Option<BitRangeInfo> {
                (id == crate::BitRangeFieldId::new(0)).then_some(BitRangeInfo {
                    storage: Varnode::new(SpaceId::new(3), 0, 4),
                    start: 3,
                    size: 5,
                })
            }
            fn address_size(&self, _space: SpaceId) -> Option<usize> {
                Some(4)
            }
        }

        let read = lower_instruction(
            &ast(vec![AstNode::Assignment {
                lhs: Ident::Named(LocalVarId(0)),
                size: None,
                rhs: ExpressionTy::Ident(Ident::BitRange(crate::BitRangeFieldId::new(0)))
                    .with_size(1),
            }]),
            &BitRangeContext,
        )
        .unwrap();
        assert_eq!(
            read.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
            vec![Opcode::IntRight, Opcode::IntAnd, Opcode::SubPiece]
        );
        assert_eq!(read.ops[0].inputs[1], Varnode::constant(3, 4));
        assert_eq!(read.ops[1].inputs[1], Varnode::constant(0x1f, 4));

        let write = lower_instruction(
            &ast(vec![AstNode::Assignment {
                lhs: Ident::BitRange(crate::BitRangeFieldId::new(0)),
                size: None,
                rhs: int(0xff, 1),
            }]),
            &BitRangeContext,
        )
        .unwrap();
        assert_eq!(
            write.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
            vec![
                Opcode::IntZext,
                Opcode::IntAnd,
                Opcode::IntLeft,
                Opcode::IntAnd,
                Opcode::IntOr,
            ]
        );
        assert_eq!(
            write.ops.last().unwrap().output.unwrap().space,
            SpaceId::new(3)
        );
        assert_eq!(write.ops.last().unwrap().output.unwrap().offset, 0);
    }

    #[test]
    fn lower_binary_operations_coerce_narrow_operands() {
        struct MixedWidthContext;
        impl PcodeLoweringContext for MixedWidthContext {
            fn default_space(&self) -> SpaceId {
                SpaceId::new(1)
            }
            fn unique_space(&self) -> SpaceId {
                SpaceId::new(2)
            }
            fn register_varnode(&self, id: RegisterId) -> Option<Varnode> {
                Some(Varnode::new(
                    SpaceId::new(3),
                    usize::from(id) as u64 * 2,
                    if id == RegisterId::new(0) { 2 } else { 1 },
                ))
            }
            fn bitrange_info(&self, _id: crate::BitRangeFieldId) -> Option<BitRangeInfo> {
                None
            }
            fn address_size(&self, _space: SpaceId) -> Option<usize> {
                Some(8)
            }
        }

        let pcode = lower_instruction(
            &ast(vec![AstNode::Assignment {
                lhs: Ident::Register(RegisterId::new(0)),
                size: None,
                rhs: ExpressionTy::Binop(Binop {
                    op: BinaryOperator::LeftShift,
                    lhs: Box::new(
                        ExpressionTy::Ident(Ident::Register(RegisterId::new(0))).with_size(2),
                    ),
                    rhs: Box::new(
                        ExpressionTy::Ident(Ident::Register(RegisterId::new(1))).with_size(1),
                    ),
                })
                .with_size(2),
            }]),
            &MixedWidthContext,
        )
        .unwrap();
        assert_eq!(pcode.ops[0].opcode, Opcode::IntZext);
        assert_eq!(pcode.ops[0].output.unwrap().size, 2);
        assert_eq!(pcode.ops[1].opcode, Opcode::IntLeft);
        assert_eq!(pcode.ops[1].inputs[1].size, 2);
    }

    #[test]
    fn lower_store_coerces_its_value_to_the_declared_width() {
        let pcode = lower_instruction(
            &ast(vec![AstNode::LoadAssignment {
                lhs: Load {
                    space: None,
                    size: Some(1),
                    ptr: Box::new(int(0, 8)),
                },
                size: None,
                rhs: ident(RegisterId::new(0)),
            }]),
            &Context,
        )
        .unwrap();
        assert_eq!(pcode.ops[0].opcode, Opcode::SubPiece);
        assert_eq!(pcode.ops[0].output.unwrap().size, 1);
        assert_eq!(pcode.ops[1].opcode, Opcode::Store);
        assert_eq!(pcode.ops[1].inputs[2].size, 1);
    }

    #[test]
    fn lower_range_assignment_supplies_its_rhs_width_to_a_userop() {
        let pcode = lower_instruction(
            &ast(vec![AstNode::RangeAssignment {
                lhs: Range {
                    value: Box::new(ident(RegisterId::new(0))),
                    start: RangeParam::Literal(0),
                    size: RangeParam::Literal(8),
                },
                size: None,
                rhs: Expression {
                    ty: ExpressionTy::PcodeOp {
                        id: PCodeOpId::new(7),
                        args: vec![],
                    },
                    size: None,
                    span: (),
                },
            }]),
            &Context,
        )
        .unwrap();
        let userop = pcode
            .ops
            .iter()
            .find(|op| op.opcode == Opcode::CallOther)
            .expect("range-assignment user-op was emitted");
        assert_eq!(userop.output.unwrap().size, 1);
    }

    #[test]
    fn lower_rejects_nodes_that_are_not_final_raw_pcode() {
        let error = lower_instruction(&ast(vec![AstNode::Build(crate::TableId::new(0))]), &Context)
            .unwrap_err();
        assert_eq!(error, PcodeLowerError::InternalNode("build statement"));
        let error = lower_instruction(
            &ast(vec![AstNode::Branch {
                target: LabelOrNode::Node("unresolved".into()),
            }]),
            &Context,
        )
        .unwrap_err();
        assert_eq!(
            error,
            PcodeLowerError::InternalNode("unresolved branch target")
        );
    }

    #[test]
    fn lower_every_binary_operator_uses_its_raw_opcode() {
        use BinaryOperator::*;
        let cases = [
            (Mul, Opcode::IntMult, false),
            (Div, Opcode::IntDiv, false),
            (SignedDiv, Opcode::IntSDiv, false),
            (Mod, Opcode::IntRem, false),
            (SignedMod, Opcode::IntSRem, false),
            (FloatDiv, Opcode::FloatDiv, false),
            (FloatMul, Opcode::FloatMult, false),
            (Add, Opcode::IntAdd, false),
            (Sub, Opcode::IntSub, false),
            (FloatAdd, Opcode::FloatAdd, false),
            (FloatSub, Opcode::FloatSub, false),
            (LeftShift, Opcode::IntLeft, false),
            (RightShift, Opcode::IntRight, false),
            (SignedRightShift, Opcode::IntSRight, false),
            (SignedLessThan, Opcode::IntSLess, false),
            (SignedGreaterThan, Opcode::IntSLess, true),
            (SignedLessEqual, Opcode::IntSLessEqual, false),
            (SignedGreaterEqual, Opcode::IntSLessEqual, true),
            (LessEqual, Opcode::IntLessEqual, false),
            (GreaterEqual, Opcode::IntLessEqual, true),
            (LessThan, Opcode::IntLess, false),
            (GreaterThan, Opcode::IntLess, true),
            (FloatLessEqual, Opcode::FloatLessEqual, false),
            (FloatGreaterEqual, Opcode::FloatLessEqual, true),
            (FloatLessThan, Opcode::FloatLess, false),
            (FloatGreaterThan, Opcode::FloatLess, true),
            (Equal, Opcode::IntEqual, false),
            (NotEqual, Opcode::IntNotEqual, false),
            (FloatEqual, Opcode::FloatEqual, false),
            (FloatNotEqual, Opcode::FloatNotEqual, false),
            (LogicalXor, Opcode::BoolXor, false),
            (LogicalAnd, Opcode::BoolAnd, false),
            (LogicalOr, Opcode::BoolOr, false),
            (BitwiseXor, Opcode::IntXor, false),
            (BitwiseOr, Opcode::IntOr, false),
            (BitwiseAnd, Opcode::IntAnd, false),
        ];
        for (operator, opcode, reverse) in cases {
            assert_eq!(super::binary_opcode(operator), (opcode, reverse));
        }
    }

    #[test]
    fn lower_every_builtin_and_unary_operator() {
        let builtins = [
            (crate::Builtin::Carry, Opcode::IntCarry),
            (crate::Builtin::Scarry, Opcode::IntSCarry),
            (crate::Builtin::Sborrow, Opcode::IntSBorrow),
            (crate::Builtin::Nan, Opcode::FloatNan),
            (crate::Builtin::Abs, Opcode::FloatAbs),
            (crate::Builtin::Sqrt, Opcode::FloatSqrt),
            (crate::Builtin::Floor, Opcode::FloatFloor),
            (crate::Builtin::Ceil, Opcode::FloatCeil),
            (crate::Builtin::Round, Opcode::FloatRound),
            (crate::Builtin::Int2Float, Opcode::FloatInt2Float),
            (crate::Builtin::Float2Float, Opcode::FloatFloat2Float),
            (crate::Builtin::Trunc, Opcode::FloatTrunc),
            (crate::Builtin::Zext, Opcode::IntZext),
            (crate::Builtin::Sext, Opcode::IntSext),
            (crate::Builtin::Popcount, Opcode::PopCount),
            (crate::Builtin::Lzcount, Opcode::LzCount),
            (crate::Builtin::Cpool, Opcode::CpoolRef),
            (crate::Builtin::NewObject, Opcode::New),
        ];
        for (builtin, opcode) in builtins {
            let expression = Expression {
                ty: ExpressionTy::FunctionCall {
                    builtin,
                    args: vec![int(1, 4)],
                },
                size: Some(4),
                span: (),
            };
            let pcode = lower_instruction(
                &ast(vec![AstNode::Assignment {
                    lhs: Ident::Named(LocalVarId(0)),
                    size: None,
                    rhs: expression,
                }]),
                &Context,
            )
            .unwrap();
            assert_eq!(pcode.ops[0].opcode, opcode);
        }
        for (operator, opcode) in [
            (crate::UnaryOperator::LogicalNot, Opcode::BoolNegate),
            (crate::UnaryOperator::BitwiseNot, Opcode::IntNegate),
            (crate::UnaryOperator::Minus, Opcode::Int2Comp),
            (crate::UnaryOperator::FloatMinus, Opcode::FloatNeg),
        ] {
            let expression = Expression {
                ty: ExpressionTy::Unop(crate::Unop {
                    op: operator,
                    e: Box::new(ident(RegisterId::new(0))),
                }),
                size: Some(4),
                span: (),
            };
            let pcode = lower_instruction(
                &ast(vec![AstNode::Assignment {
                    lhs: Ident::Named(LocalVarId(0)),
                    size: None,
                    rhs: expression,
                }]),
                &Context,
            )
            .unwrap();
            assert_eq!(pcode.ops[0].opcode, opcode);
        }
    }

    #[test]
    fn lower_direct_and_indirect_control_flow() {
        let direct = lower_instruction(
            &ast(vec![AstNode::Call {
                target: LabelOrNode::Expr(int(0x1000, 8)),
            }]),
            &Context,
        )
        .unwrap();
        assert_eq!(
            direct.ops[0],
            PcodeOp::new(
                Opcode::Call,
                None,
                vec![Varnode::new(SpaceId::new(1), 0x1000, 8)]
            )
        );
        let indirect = lower_instruction(
            &ast(vec![
                AstNode::ConditionalBranch {
                    condition: int(1, 1),
                    target: LabelOrNode::Expr(int(0x2000, 8)),
                },
                AstNode::BranchIndirect {
                    target: ident(RegisterId::new(0)),
                },
                AstNode::CallIndirect {
                    target: ident(RegisterId::new(1)),
                },
                AstNode::Return {
                    target: ident(RegisterId::new(2)),
                },
            ]),
            &Context,
        )
        .unwrap();
        assert_eq!(
            indirect.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
            vec![
                Opcode::CBranch,
                Opcode::BranchInd,
                Opcode::CallInd,
                Opcode::Return
            ]
        );
        assert_eq!(
            indirect.ops[0].inputs,
            vec![
                Varnode::new(SpaceId::new(1), 0x2000, 8),
                Varnode::constant(1, 1)
            ]
        );
    }

    #[test]
    fn lowering_errors_are_displayable_and_typed() {
        let errors = [
            PcodeLowerError::UnknownSize,
            PcodeLowerError::ZeroSize,
            PcodeLowerError::CopySizeMismatch {
                input: 1,
                output: 2,
            },
            PcodeLowerError::InputSizeMismatch {
                operation: "operation",
                left: 1,
                right: 2,
            },
            PcodeLowerError::InvalidBooleanSize(2),
            PcodeLowerError::AddressSizeMismatch {
                expected: 4,
                actual: 8,
            },
            PcodeLowerError::StoreSizeMismatch {
                declared: 4,
                value: 8,
            },
            PcodeLowerError::InvalidRange {
                start: 0,
                size: 0,
                storage_bits: 32,
            },
            PcodeLowerError::UniqueSpaceOverflow,
            PcodeLowerError::UnknownRegister(RegisterId::new(9)),
            PcodeLowerError::UnresolvedIdentifier("field"),
            PcodeLowerError::UnresolvedSpace,
            PcodeLowerError::UnresolvedRangeParameter,
            PcodeLowerError::InternalNode("macro"),
            PcodeLowerError::Unsupported("range"),
            PcodeLowerError::DuplicateLabel("loop".into()),
            PcodeLowerError::UnknownLabel("loop".into()),
            PcodeLowerError::InvalidDirectTarget,
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
        assert_eq!(
            InstructionPcode::lower(&ast(vec![]), &Context).unwrap(),
            InstructionPcode::new()
        );
    }

    #[test]
    fn lower_rejects_invalid_raw_widths_and_ranges() {
        let range = |start, size| Expression {
            ty: ExpressionTy::Range(crate::Range {
                value: Box::new(ident(RegisterId::new(0))),
                start: crate::RangeParam::Literal(start),
                size: crate::RangeParam::Literal(size),
            }),
            size: None,
            span: (),
        };
        for (start, size) in [(0, 0), (0, 65), (31, 2)] {
            let error = lower_instruction(
                &ast(vec![AstNode::Assignment {
                    lhs: Ident::Register(RegisterId::new(0)),
                    size: None,
                    rhs: range(start, size),
                }]),
                &Context,
            )
            .unwrap_err();
            assert!(matches!(error, PcodeLowerError::InvalidRange { .. }));
        }

        let error = lower_instruction(
            &ast(vec![AstNode::Assignment {
                lhs: Ident::Named(LocalVarId(0)),
                size: None,
                rhs: Expression {
                    ty: ExpressionTy::Range(crate::Range {
                        value: Box::new(ident(RegisterId::new(0))),
                        start: crate::RangeParam::MacroArg(LocalVarId(1)),
                        size: crate::RangeParam::Literal(1),
                    }),
                    size: None,
                    span: (),
                },
            }]),
            &Context,
        )
        .unwrap_err();
        assert_eq!(error, PcodeLowerError::UnresolvedRangeParameter);

        let mismatch = ExpressionTy::Binop(Binop {
            op: BinaryOperator::Add,
            lhs: Box::new(ident(RegisterId::new(0))),
            rhs: Box::new(int(1, 1)),
        })
        .with_size(4);
        // Integer literals are sized by their consuming p-code operation,
        // rather than forcing a mixed-width raw operation.
        assert!(
            lower_instruction(
                &ast(vec![AstNode::Assignment {
                    lhs: Ident::Named(LocalVarId(0)),
                    size: None,
                    rhs: mismatch,
                }]),
                &Context,
            )
            .is_ok()
        );

        let comparison = ExpressionTy::Binop(Binop {
            op: BinaryOperator::Equal,
            lhs: Box::new(ident(RegisterId::new(0))),
            rhs: Box::new(ident(RegisterId::new(1))),
        })
        .with_size(4);
        assert_eq!(
            lower_instruction(
                &ast(vec![AstNode::Assignment {
                    lhs: Ident::Named(LocalVarId(0)),
                    size: None,
                    rhs: comparison,
                }]),
                &Context,
            )
            .unwrap_err(),
            PcodeLowerError::InvalidBooleanSize(4)
        );

        let bad_load = Load {
            space: Some(PcodeSpaceRef::Resolved(SpaceId::new(1))),
            size: Some(4),
            ptr: Box::new(ident(RegisterId::new(0))),
        };
        assert!(matches!(
            lower_instruction(
                &ast(vec![AstNode::Assignment {
                    lhs: Ident::Named(LocalVarId(0)),
                    size: Some(4),
                    rhs: ExpressionTy::Load(bad_load.clone()).with_size(4),
                }]),
                &Context,
            ),
            Err(PcodeLowerError::AddressSizeMismatch { .. })
        ));
        assert!(matches!(
            lower_instruction(
                &ast(vec![AstNode::LoadAssignment {
                    lhs: bad_load,
                    size: None,
                    rhs: int(1, 1),
                }]),
                &Context,
            ),
            Err(PcodeLowerError::AddressSizeMismatch { .. })
        ));
    }

    #[test]
    fn lower_handles_subpieces_and_rejects_invalid_final_forms() {
        let lsb = Expression {
            ty: ExpressionTy::SubPieceLsb {
                src: Box::new(ident(RegisterId::new(0))),
                count: 2,
            },
            size: Some(2),
            span: (),
        };
        let msb = Expression {
            ty: ExpressionTy::SubPieceMsb {
                src: Box::new(ident(RegisterId::new(0))),
                count: 2,
            },
            size: Some(2),
            span: (),
        };
        let pcode = lower_instruction(
            &ast(vec![
                AstNode::Assignment {
                    lhs: Ident::Named(LocalVarId(0)),
                    size: None,
                    rhs: lsb,
                },
                AstNode::Assignment {
                    lhs: Ident::Named(LocalVarId(1)),
                    size: None,
                    rhs: msb,
                },
            ]),
            &Context,
        )
        .unwrap();
        assert_eq!(
            pcode.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
            vec![Opcode::SubPiece, Opcode::SubPiece]
        );
        assert_eq!(pcode.ops[0].inputs[1], Varnode::constant(0, 8));
        assert_eq!(pcode.ops[1].inputs[1], Varnode::constant(2, 8));

        let range = Expression {
            ty: ExpressionTy::Range(crate::Range {
                value: Box::new(ident(RegisterId::new(0))),
                start: crate::RangeParam::Literal(0),
                size: crate::RangeParam::Literal(1),
            }),
            size: Some(1),
            span: (),
        };
        let pcode = lower_instruction(
            &ast(vec![AstNode::Assignment {
                lhs: Ident::Named(LocalVarId(0)),
                size: None,
                rhs: range,
            }]),
            &Context,
        )
        .unwrap();
        assert_eq!(
            pcode.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
            vec![Opcode::IntRight, Opcode::IntAnd, Opcode::SubPiece]
        );
        let error = lower_instruction(
            &ast(vec![
                AstNode::Label("same".into()),
                AstNode::Label("same".into()),
            ]),
            &Context,
        )
        .unwrap_err();
        assert_eq!(error, PcodeLowerError::DuplicateLabel("same".into()));
        let error = lower_instruction(
            &ast(vec![AstNode::Branch {
                target: LabelOrNode::Label("missing".into()),
            }]),
            &Context,
        )
        .unwrap_err();
        assert_eq!(error, PcodeLowerError::UnknownLabel("missing".into()));
        let error = lower_instruction(
            &ast(vec![AstNode::Branch {
                target: LabelOrNode::Expr(ident(RegisterId::new(0))),
            }]),
            &Context,
        )
        .unwrap_err();
        assert_eq!(error, PcodeLowerError::InvalidDirectTarget);
    }
}
