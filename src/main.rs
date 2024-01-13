use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::OptimizationLevel;
use inkwell::values::AnyValue;

use std::error::Error;

type SumFunc = unsafe extern "C" fn(u64) -> u64;

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn jit_compile_sum(&self) -> Option<JitFunction<SumFunc>> {
        let i64_type = self.context.i64_type();
        
        let main_type = i64_type.fn_type(&[i64_type.into()], false);
        let main = self.module.add_function("main", main_type, None);
        let init_block = self.context.append_basic_block(main, "entry");
        self.builder.position_at_end(init_block);
        let vec = self.builder.build_array_malloc(i64_type, i64_type.const_int(100000000000, false), "vec").unwrap();
        self.builder.build_memset(vec, 8, i64_type.const_zero(), i64_type.const_int(200,false)).unwrap();
        let x = main.get_nth_param(0)?.into_int_value();
        let fn_type =  i64_type.fn_type(&[i64_type.into(), vec.get_type().into()], false);
        let function = self.module.add_function("sum", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "en");
        let callres = self.builder.build_call(function, &[x.into(), vec.into()], "call").unwrap().try_as_basic_value().left().unwrap().as_any_value_enum().into_int_value();
        self.builder.build_return(Some(&callres)).unwrap();
       
        self.builder.position_at_end(basic_block); 
        
        let x = function.get_nth_param(0)?.into_int_value();
        let y = function.get_nth_param(1)?.into_pointer_value();
        let one = i64_type.const_int(1,false);
        let two = i64_type.const_int(2,false);
        let is_one = self.builder.build_int_compare(inkwell::IntPredicate::ULE, x, one, "isone").unwrap();
        let rtn1 = self.context.append_basic_block(function, "rtn1"); 
        let rtn2 = self.context.append_basic_block(function, "rtn2");
        let rtn3 = self.context.append_basic_block(function, "rtn3");
        let rtn4 = self.context.append_basic_block(function, "rnt4");

        self.builder.build_conditional_branch(is_one, rtn1, rtn2).unwrap();
        
        self.builder.position_at_end(rtn1);
        self.builder.build_return(Some(&x)).unwrap();
        
        
        self.builder.position_at_end(rtn2);
        let ptr = unsafe { self.builder.build_gep(y, &[x], "gep").unwrap() };
        let val = self.builder.build_load(ptr, "load").unwrap().into_int_value();
        let cmp = self.builder.build_int_compare(inkwell::IntPredicate::EQ, val, i64_type.const_zero(), "iszero").unwrap();
        self.builder.build_conditional_branch(cmp, rtn4, rtn3).unwrap();

        self.builder.position_at_end(rtn3);
        let ptr = unsafe { self.builder.build_gep(y, &[x], "gep").unwrap() };
        let val = self.builder.build_load(ptr, "load").unwrap().into_int_value();
        self.builder.build_return(Some(&val)).unwrap();
        
        self.builder.position_at_end(rtn4);
        let sub1 = self.builder.build_int_sub(x, one, "sub").unwrap();
        let sub2 = self.builder.build_int_sub(x, two, "sub").unwrap();
        let c1 = self.builder.build_call(function, &[sub1.into(), y.into()], "call").unwrap();
        let c2 = self.builder.build_call(function, &[sub2.into(), y.into()], "call").unwrap();
        let v1 = c1.try_as_basic_value().left().unwrap().as_any_value_enum().into_int_value();
        let v2 = c2.try_as_basic_value().left().unwrap().as_any_value_enum().into_int_value();
        
        let sum = self.builder.build_int_add(v1, v2, "sum").unwrap();

        let ptr = unsafe { self.builder.build_gep(y, &[x], "gep").unwrap() };
        self.builder.build_store(ptr, sum).unwrap();
        
        self.builder.build_return(Some(&sum)).unwrap();
        
        //self.builder.build_return(Some(&val)).unwrap();
        
        unsafe { self.execution_engine.get_function("main").ok() }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = context.create_module("sum");
    
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive)?;
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    let sum = codegen.jit_compile_sum().ok_or("Unable to JIT compile `sum`")?;

    let args : Vec<String> = std::env::args().collect();
    let x = args[1].parse::<u64>().unwrap();
    let timer = std::time::Instant::now();
   
    unsafe {
        println!("{}", sum.call(x));
    }

    println!("{}", timer.elapsed().as_secs());
    Ok(())
}