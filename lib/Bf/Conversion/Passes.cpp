#include "Bf/Conversion/Passes.h"

#include "Bf/Dialect/bf/IR/BfDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Bf/Dialect/bf/IR/BfOps.h"
#include "Bf/Dialect/bflow/IR/BfLowDialect.h"
#include "Bf/Dialect/bflow/IR/BfLowOps.h"
#include <cstdint>
#include <iterator>
#include <iostream>
#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/IR/Block.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <utility>

namespace bf {
#define GEN_PASS_DEF_BFCOMBINECONSECUTIVEOPS
#include "Bf/Conversion/Passes.h.inc"

class ConsecutiveShiftPattern : public mlir::OpRewritePattern<bf::Shift> {
public:
    using mlir::OpRewritePattern<bf::Shift>::OpRewritePattern;

    void initialize() {
        setHasBoundedRewriteRecursion();
    }

    llvm::LogicalResult matchAndRewrite(bf::Shift op, mlir::PatternRewriter &rewriter) const final {
        mlir::Operation *nextOp = op->getNextNode();
        if (!nextOp) {
            return llvm::failure();
        }

        bf::Shift nextDialectOp = mlir::dyn_cast<bf::Shift>(nextOp);
        if (!nextDialectOp) {
            return llvm::failure();
        }

        int a1 = op.getAmount();
        int a2 = nextDialectOp.getAmount();
        
        rewriter.setInsertionPoint(op);

        rewriter.eraseOp(nextDialectOp);
        rewriter.modifyOpInPlace(op, [&op, a1, a2]() {op.setAmount(a1 + a2); });
        return llvm::success();
    }
};

class ConsecutiveIncrementPattern : public mlir::OpRewritePattern<bf::Increment> {
public:
    using mlir::OpRewritePattern<bf::Increment>::OpRewritePattern;

    void initialize() {
        setHasBoundedRewriteRecursion();
    }

    llvm::LogicalResult matchAndRewrite(bf::Increment op, mlir::PatternRewriter &rewriter) const final {
        mlir::Operation *nextOp = op->getNextNode();
        if (!nextOp) {
            return llvm::failure();
        }

        bf::Increment nextDialectOp = mlir::dyn_cast<bf::Increment>(nextOp);
        if (!nextDialectOp) {
            return llvm::failure();
        }

        int16_t a1 = op.getAmount();
        int16_t a2 = nextDialectOp.getAmount();

        // if ((int) a1 + (int) a2 > 127) {
        //     return llvm::failure();
        // } 

        // if ((int) a1 + (int) a2 < -126) {
        //     return llvm::failure();
        // }

        rewriter.setInsertionPoint(op);
        rewriter.eraseOp(nextDialectOp);
        rewriter.modifyOpInPlace(op, [&op, a1, a2]() {op.setAmount(a1 + a2); });
        return llvm::success();
    }
};

class BfCombineConsecutiveOps : public impl::BfCombineConsecutiveOpsBase<BfCombineConsecutiveOps> {
public:
    using impl::BfCombineConsecutiveOpsBase<BfCombineConsecutiveOps>::BfCombineConsecutiveOpsBase;

    void runOnOperation() final {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<ConsecutiveShiftPattern, ConsecutiveIncrementPattern>(&getContext());
        mlir::FrozenRewritePatternSet patternSet(std::move(patterns));

        if (llvm::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
            signalPassFailure();
        }
    }
};

#define GEN_PASS_DEF_BFSETZERO
#include "Bf/Conversion/Passes.h.inc"

class SetZeroPattern : public mlir::OpRewritePattern<bf::Loop> {
public:
    using mlir::OpRewritePattern<bf::Loop>::OpRewritePattern;

    llvm::LogicalResult matchAndRewrite(bf::Loop op, mlir::PatternRewriter &rewriter) const final {
        mlir::Region &region = op.getBody();
        if (!region.hasOneBlock()) {
            return llvm::failure();
        }
        mlir::Block &block = region.front();
        auto b = block.begin();
        auto e = block.end();
        auto l = std::distance(b, e);
        if (l != 1) {
            return llvm::failure();
        }
        mlir::Operation &o = block.front();
        auto increment = llvm::dyn_cast<bf::Increment>(o);
        if (!increment) {
            return llvm::failure();
        }
        if (increment.getAmount() == 1 || increment.getAmount() == -1) {
            auto sz = rewriter.create<bf::SetZero>(op.getLoc(), 0);
            rewriter.replaceOp(op, sz);
            return llvm::success();
        } else {
            return llvm::failure();
        }
    }
};

class BfSetZero: public impl::BfSetZeroBase<BfSetZero> {
public:
    using impl::BfSetZeroBase<BfSetZero>::BfSetZeroBase;

    void runOnOperation() final {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<SetZeroPattern>(&getContext());
        mlir::FrozenRewritePatternSet patternSet(std::move(patterns));

        if (llvm::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
            signalPassFailure();
        }
    }
};

#define GEN_PASS_DEF_BFOFFSET
#include "Bf/Conversion/Passes.h.inc"

class OffsetPattern : public mlir::OpRewritePattern<mlir::func::FuncOp> {
public:
    using mlir::OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

    void initialize() {
        setHasBoundedRewriteRecursion();
    }

    llvm::LogicalResult matchAndRewrite(mlir::func::FuncOp op, mlir::PatternRewriter &rewriter) const final {
        // Would look like >>>>++>+++>>+++ -> +(2, 4) +(3, 5) +(3, 7) >(7)
        if (op->hasAttr("already_optimized")) {
            std::cout << "Failing here" << std::endl;
            return llvm::failure();
        }
        std::cout << "Doesnt fail" << std::endl;
        mlir::Region &region = op.getBody();
        if (!region.hasOneBlock()) {
            return llvm::failure();
        }
        mlir::Block &block = region.front();
        
        auto res = helper(block, rewriter);
        if (res.succeeded()) {
            op->setAttr("already_optimized", rewriter.getUnitAttr());
            return llvm::success();
        } else {
            return llvm::failure();
        }
    }
private:
    llvm::LogicalResult helper(mlir::Block &block, mlir::PatternRewriter &rewriter) const {
        int offset = 0;

        llvm::SmallVector<mlir::Operation*> operations;
        for (mlir::Operation &op : block.getOperations()) {
            operations.push_back(&op);
        }
        
        for (mlir::Operation *opPtr : operations) {
            mlir::Operation &op = *opPtr;

            if (op.getBlock() != &block) {
                continue;
            }
            
            bf::Increment inc = mlir::dyn_cast<bf::Increment>(op);
            if (inc) {
                rewriter.setInsertionPoint(inc);
                rewriter.create<bf::Increment>(inc.getLoc(), inc.getAmount(), offset);
                rewriter.eraseOp(inc);
                continue;
            }
            bf::Shift shift = mlir::dyn_cast<bf::Shift>(op);
            if (shift) {
                offset += shift.getAmount();
                rewriter.eraseOp(shift);
                continue;
            }
            bf::SetZero sz = mlir::dyn_cast<bf::SetZero>(op);
            if (sz) {
                rewriter.setInsertionPoint(sz);
                rewriter.create<bf::SetZero>(sz.getLoc(), offset);
                rewriter.eraseOp(sz);
                continue;
            }
            // Add multiply op here later, to optimize that as well
            /**
            bf::Loop loop = mlir::dyn_cast<bf::Loop>(op);
            if (loop) {
                rewriter.setInsertionPoint(loop);
                if (offset != 0) {
                    rewriter.create<bf::Shift>(loop.getLoc(), offset);
                    offset = 0;
                }
                
                mlir::Region &region = loop.getBody();
                if (!region.hasOneBlock()) {
                    return llvm::failure();
                }
                mlir::Block &block = region.front();
                llvm::LogicalResult res = helper(block, rewriter);
                if (res.failed()) {
                    return res;
                }

                continue;
            }
            */
            
            if (mlir::isa<bf::Input, bf::Output, bf::Mul, bf::Loop>(op)) {
                if (offset != 0) {
                    rewriter.setInsertionPoint(&op);
                    rewriter.create<bf::Shift>(op.getLoc(), offset);
                    offset = 0;
                }                                         
            }
        
        }
        if (offset != 0) {
            if (block.empty()) {
                rewriter.setInsertionPointToEnd(&block);
            } else if (mlir::dyn_cast<mlir::func::ReturnOp>(block.back())) {
                rewriter.setInsertionPoint(&block.back());
            } else {
                rewriter.setInsertionPointToEnd(&block);
            }
            rewriter.create<bf::Shift>(rewriter.getUnknownLoc(), offset);
        }
        return llvm::success();
    }
};

class BfOffset: public impl::BfOffsetBase<BfOffset> {
public:
    using impl::BfOffsetBase<BfOffset>::BfOffsetBase;

    void runOnOperation() final {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<OffsetPattern>(&getContext());
        mlir::FrozenRewritePatternSet patternSet(std::move(patterns));

        if (llvm::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
            signalPassFailure();
        }
    }
};

#define GEN_PASS_DEF_BFMULLOOP
#include "Bf/Conversion/Passes.h.inc"

class MulLoopPattern : public mlir::OpRewritePattern<bf::Loop> {
public:
    using mlir::OpRewritePattern<bf::Loop>::OpRewritePattern;

    llvm::LogicalResult matchAndRewrite(bf::Loop op, mlir::PatternRewriter &rewriter) const final {
        // Matches Multiply loops of the form [->+++>+++++<<] does not support [-<+++<++++>>>++++<]
        mlir::Region &region = op.getBody();
        if (!region.hasOneBlock()) {
            return llvm::failure();
        }
        mlir::Block &block = region.front();
    
        bool hasDec = false;
        for (mlir::Operation &o : block.getOperations()) {
            bf::Increment inc = mlir::dyn_cast<bf::Increment>(o);
            if (!inc) {
                return llvm::failure();
            }
            if (inc.getAmount() == -1 && inc.getOffset() == 0) {
                hasDec = true;
            } else {
                if (inc.getAmount() < 1) {
                    return llvm::failure();
                }
                rewriter.setInsertionPoint(op);
                rewriter.create<bf::Mul>(op.getLoc(), inc.getOffset(), inc.getAmount());
            }
        }
        rewriter.setInsertionPoint(op);
        rewriter.create<bf::SetZero>(op.getLoc(), 0);

        if (hasDec) {
            rewriter.eraseOp(op);
            return llvm::success();
        } else {
            return llvm::failure();
        }
    }
};

class BfMulLoop: public impl::BfMulLoopBase<BfMulLoop> {
public:
    using impl::BfMulLoopBase<BfMulLoop>::BfMulLoopBase;

    void runOnOperation() final {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<MulLoopPattern>(&getContext());
        mlir::FrozenRewritePatternSet patternSet(std::move(patterns));

        if (llvm::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
            signalPassFailure();
        }
    }
};

#define GEN_PASS_DEF_BFTOBFLOW
#include "Bf/Conversion/Passes.h.inc"

class BfIncrementLowering : public mlir::ConversionPattern {
public:
    BfIncrementLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bf::Increment::getOperationName(), 1, ctx) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto incrementOp = llvm::dyn_cast<bf::Increment>(op);
        if (!incrementOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        auto i8Type = rewriter.getI16Type();

        auto rp = rewriter.create<bflow::read_ptr>(loc, i8Type, incrementOp.getOffset());
        auto constant_attr = rewriter.getIntegerAttr(i8Type, incrementOp.getAmount());
        auto amount = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, constant_attr);
        auto res = rewriter.create<mlir::arith::AddIOp>(loc, rp, amount);

        auto wp = rewriter.create<bflow::write_ptr>(loc, res.getResult(), incrementOp.getOffset());
        rewriter.replaceOp(op, wp);
        return llvm::success();
    }
};

class BfShiftLowering : public mlir::ConversionPattern {
public:
    BfShiftLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bf::Shift::getOperationName(), 1, ctx) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto shiftOp = llvm::dyn_cast<bf::Shift>(op);
        if (!shiftOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        auto nShift = rewriter.create<bflow::shift_ptr>(loc, shiftOp.getAmountAttr());
        rewriter.replaceOp(op, nShift);
        return llvm::success();
    }
};

class BfInputLowering : public mlir::ConversionPattern {
public:
    BfInputLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bf::Input::getOperationName(), 1, ctx) {}
    
    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto inputOp = llvm::dyn_cast<bf::Input>(op);
        if (!inputOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

        auto getCharRef = getOrInsertGetChar(rewriter, parentModule);
        auto i8Type = rewriter.getI16Type();

        auto callOp = rewriter.create<mlir::func::CallOp>(loc, getCharRef, rewriter.getI32Type());
        auto trunc = rewriter.create<mlir::LLVM::TruncOp>(loc, i8Type, callOp.getResult(0));
        auto wp = rewriter.create<bflow::write_ptr>(loc, trunc, 0);
        rewriter.replaceOp(op, wp);
        return llvm::success();
    }

private:
    static mlir::FlatSymbolRefAttr getOrInsertGetChar(mlir::ConversionPatternRewriter &rewriter, mlir::ModuleOp module) {
        mlir::MLIRContext *ctx = module.getContext();
        if (module.lookupSymbol<mlir::func::FuncOp>("getchar")) {
            return mlir::SymbolRefAttr::get(ctx, "getchar");
        }

        mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto consoleType = rewriter.getI32Type();
        rewriter.create<mlir::func::FuncOp>(module.getLoc(),
                                            "getchar",
                                            rewriter.getFunctionType(mlir::TypeRange(), consoleType),
                                            rewriter.getStringAttr("private"),
                                            mlir::ArrayAttr(),
                                            mlir::ArrayAttr()
                                        );
        return mlir::SymbolRefAttr::get(ctx, "getchar");
    }
};

class BfOutputLowering : public mlir::ConversionPattern {
public:
    BfOutputLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bf::Output::getOperationName(), 1, ctx) {}
    
    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto outputOp = llvm::dyn_cast<bf::Output>(op);
        if (!outputOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

        auto putCharRef = getOrInsertPutChar(rewriter, parentModule);
        auto i8Type = rewriter.getI16Type();

        auto rp = rewriter.create<bflow::read_ptr>(loc, i8Type, 0);
        auto cast = rewriter.create<mlir::LLVM::SExtOp>(loc, rewriter.getI32Type(), rp);
        rewriter.create<mlir::func::CallOp>(loc, putCharRef, mlir::TypeRange(rewriter.getI32Type()), mlir::ValueRange(cast));
        rewriter.eraseOp(op);
        
        return llvm::success();
    }

private:
    static mlir::FlatSymbolRefAttr getOrInsertPutChar(mlir::ConversionPatternRewriter &rewriter, mlir::ModuleOp module) {
        mlir::MLIRContext *ctx = module.getContext();
        if (module.lookupSymbol<mlir::func::FuncOp>("putchar")) {
            return mlir::SymbolRefAttr::get(ctx, "putchar");
        }

        mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto consoleType = rewriter.getI32Type();
        rewriter.create<mlir::func::FuncOp>(module.getLoc(),
                                            "putchar",
                                            rewriter.getFunctionType(consoleType, consoleType),
                                            rewriter.getStringAttr("private"),
                                            mlir::ArrayAttr(),
                                            mlir::ArrayAttr()
                                        );
        return mlir::SymbolRefAttr::get(ctx, "putchar");
    }
};

class BfLoopLowering : public mlir::ConversionPattern {
public:
    BfLoopLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bf::Loop::getOperationName(), 1, ctx) {}

    void initialize() {
        setHasBoundedRewriteRecursion();
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto loopOp = llvm::dyn_cast<bf::Loop>(op);
        if (!loopOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        auto scfWhile = rewriter.create<mlir::scf::WhileOp>(loc, op->getResultTypes(), op->getOperands());

        rewriter.createBlock(&scfWhile.getBefore());

        auto i8Type = rewriter.getI16Type();

        auto rp = rewriter.create<bflow::read_ptr>(loc, i8Type, 0);
        auto cmp = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI16IntegerAttr(0));
        auto cond = rewriter.create<mlir::arith::CmpIOp>(loc, rewriter.getI1Type(),
            mlir::arith::CmpIPredicateAttr::get(getContext(), mlir::arith::CmpIPredicate(1)),
            rp, cmp
        );
        rewriter.create<mlir::scf::ConditionOp>(loc, cond, mlir::ValueRange());

        rewriter.createBlock(&scfWhile.getAfter());
        auto &after = scfWhile.getAfter();
        auto &loopBody = loopOp.getBody();
        rewriter.cloneRegionBefore(loopBody, &after.back());
        rewriter.eraseBlock(&after.back());

        auto lastBlock = &after.back();
        if (!lastBlock->empty()) {
            auto lastOp = &lastBlock->back();
            rewriter.setInsertionPointAfter(lastOp);
        } else {
            rewriter.setInsertionPointToStart(lastBlock);
        }
        rewriter.create<mlir::scf::YieldOp>(loc);

        rewriter.replaceOp(op, scfWhile->getResults());

        return llvm::success();
    }
};

class BfSetZeroLowering : public mlir::ConversionPattern {
public:
    BfSetZeroLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bf::SetZero::getOperationName(), 1, ctx) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto szOp = llvm::dyn_cast<bf::SetZero>(op);
        if (!szOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        auto i8Type = rewriter.getI16Type();
        
        auto z = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI16IntegerAttr(0));
        auto res = rewriter.create<bflow::write_ptr>(loc, z, szOp.getOffset());
        rewriter.replaceOp(op, res);

        return llvm::success();
    }
};

class BfMulLowering : public mlir::ConversionPattern {
public:
    BfMulLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bf::Mul::getOperationName(), 1, ctx) {}
    
    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto mulOp = llvm::dyn_cast<bf::Mul>(op);
        if (!mulOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        auto i8Type = rewriter.getI16Type();

        auto rp = rewriter.create<bflow::read_ptr>(loc, i8Type, 0);
        auto mult = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, mulOp.getMultipleAttr());
        auto prod = rewriter.create<mlir::arith::MulIOp>(loc, rp, mult);
        auto dst = rewriter.create<bflow::read_ptr>(loc, i8Type, mulOp.getShift());
        auto add = rewriter.create<mlir::arith::AddIOp>(loc, dst, prod);
        auto wp = rewriter.create<bflow::write_ptr>(loc, add, mulOp.getShift());
        rewriter.replaceOp(op, wp);

        return llvm::success();
    }
};

class BfToBfLow : public impl::BfToBfLowBase<BfToBfLow> {
public:
    using impl::BfToBfLowBase<BfToBfLow>::BfToBfLowBase;

    void runOnOperation() final {
        mlir::ConversionTarget target(getContext());

        target.addLegalDialect<bflow::BfLowDialect, mlir::arith::ArithDialect, mlir::BuiltinDialect, mlir::LLVM::LLVMDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect>();
        target.addIllegalDialect<bf::BfDialect>();
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<BfIncrementLowering, BfShiftLowering, BfInputLowering, BfOutputLowering, BfLoopLowering, BfSetZeroLowering, BfMulLowering>(&getContext());
        mlir::FrozenRewritePatternSet patternSet(std::move(patterns));

        if (llvm::failed(mlir::applyFullConversion(getOperation(), target, patternSet))) {
            signalPassFailure();
        }
    }
};

#define GEN_PASS_DEF_BFLOWTOMLIR
#include "Bf/Conversion/Passes.h.inc"


mlir::memref::GetGlobalOp getOrCreateMemory(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::ModuleOp module) {
    auto mType = mlir::MemRefType::get(65535, rewriter.getI16Type());
    mlir::memref::GlobalOp memory;
    if (!(memory = module.lookupSymbol<mlir::memref::GlobalOp>("bf_memory"))) {
        mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        memory = rewriter.create<mlir::memref::GlobalOp>(loc, "bf_memory",
            rewriter.getStringAttr("private"),
            mType,
            mlir::IntegerAttr(0),
            false,
            mlir::IntegerAttr(0)    
        );
    }

    return rewriter.create<mlir::memref::GetGlobalOp>(loc, mType, llvm::StringRef("bf_memory"));
}

mlir::memref::GetGlobalOp getOrCreatePointer(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter, mlir::ModuleOp module) {
    auto mType = mlir::MemRefType::get({}, rewriter.getIndexType());
    mlir::memref::GlobalOp pointer;
    if (!(pointer = module.lookupSymbol<mlir::memref::GlobalOp>("bf_ptr"))) {
        mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
        mlir::RankedTensorType tType = mlir::RankedTensorType::get({}, rewriter.getIndexType());
        mlir::DenseElementsAttr d = mlir::DenseElementsAttr::get(tType, rewriter.getIndexAttr(0));
        rewriter.setInsertionPointToStart(module.getBody());
        pointer = rewriter.create<mlir::memref::GlobalOp>(loc, "bf_ptr",
            rewriter.getStringAttr("private"),
            mType,
            d,
            false,
            nullptr
        );
    }

    return rewriter.create<mlir::memref::GetGlobalOp>(loc, mType, llvm::StringRef("bf_ptr"));
}

class BfLowReadPtrLowering : public mlir::ConversionPattern {
public:
    BfLowReadPtrLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bflow::read_ptr::getOperationName(), 1, ctx) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto ptrOp = llvm::dyn_cast<bflow::read_ptr>(op);
        if (!ptrOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<mlir::ModuleOp>();

        auto memory = getOrCreateMemory(loc, rewriter, parentModule);
        // Read ptr here
        mlir::Value ptr = getOrCreatePointer(loc, rewriter, parentModule);
        auto index = rewriter.create<mlir::memref::LoadOp>(loc, ptr, mlir::ValueRange());
        int32_t offset = ptrOp.getOffset();
        auto offOp = rewriter.create<mlir::index::ConstantOp>(loc, offset);
        auto incr = rewriter.create<mlir::index::AddOp>(loc, index, offOp);
        auto curVal = rewriter.create<mlir::memref::LoadOp>(loc, rewriter.getI16Type(), memory, mlir::ValueRange(incr));

        rewriter.replaceOp(op, curVal);
        return llvm::success();
    }
};

class BfLowWritePtrLowering : public mlir::ConversionPattern {
public:
    BfLowWritePtrLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bflow::write_ptr::getOperationName(), 1, ctx) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto ptrOp = llvm::dyn_cast<bflow::write_ptr>(op);
        if (!ptrOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<mlir::ModuleOp>();

        auto memory = getOrCreateMemory(loc, rewriter, parentModule);
        mlir::Value ptr = getOrCreatePointer(loc, rewriter, parentModule);
        auto index = rewriter.create<mlir::memref::LoadOp>(loc, ptr, mlir::ValueRange());
        int32_t offset = ptrOp.getOffset();
        auto offOp = rewriter.create<mlir::index::ConstantOp>(loc, offset);
        auto incr = rewriter.create<mlir::index::AddOp>(loc, index, offOp);
        auto res = rewriter.create<mlir::memref::StoreOp>(loc, ptrOp.getValue(), memory, mlir::ValueRange(incr));

        rewriter.replaceOp(op, res);
        return llvm::success();
    }
};

class BfLowShiftPtrLowering : public mlir::ConversionPattern {
public:
    BfLowShiftPtrLowering(mlir::MLIRContext *ctx)
        : mlir::ConversionPattern(bflow::shift_ptr::getOperationName(), 1, ctx) {}
    
    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
        auto ptrOp = llvm::dyn_cast<bflow::shift_ptr>(op);
        if (!ptrOp) {
            return llvm::failure();
        }

        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<mlir::ModuleOp>();

        mlir::Value ptr = getOrCreatePointer(loc, rewriter, parentModule);
        auto index = rewriter.create<mlir::memref::LoadOp>(loc, ptr, mlir::ValueRange());

        int32_t amount = ptrOp.getAmount();
        auto shift = rewriter.create<mlir::index::ConstantOp>(loc, amount);
        auto incr = rewriter.create<mlir::index::AddOp>(loc, index, shift);
        auto res = rewriter.create<mlir::memref::StoreOp>(loc, incr, ptr, mlir::ValueRange());

        rewriter.replaceOp(op, res);
        return llvm::success();
    }
};

class BfLowToMLIR : public impl::BfLowToMLIRBase<BfLowToMLIR> {
public:
    using impl::BfLowToMLIRBase<BfLowToMLIR>::BfLowToMLIRBase;

    void runOnOperation() final {
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<mlir::index::IndexDialect, mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect, mlir::BuiltinDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::arith::ArithDialect>();
        target.addIllegalDialect<bflow::BfLowDialect>();

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<BfLowReadPtrLowering, BfLowWritePtrLowering, BfLowShiftPtrLowering>(&getContext());
        mlir::FrozenRewritePatternSet patternSet(std::move(patterns));

        if (llvm::failed(mlir::applyFullConversion(getOperation(), target, patternSet))) {
            signalPassFailure();
        }
    }
};
}
