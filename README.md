```python
1)
input a,b;
output y1,y2,y3,y4,y5,y6,y7;
and (y1,a,b);
or (y2,a,b);
not (y3,a);
xor (y4,a,b);
nand (y5,a,b);
nor (y6,a,b);
xnor ( y7,a,b);
endmodule

2)
module (A,B,C,D,F1);
input A,B,C,D;
output F1;
wire x1,x2,x3,x4,x5;
assign x1=(~A)&(~B)&(~C)&(~D);
assign x2=(A)&(~C)&(~D);
assign x3=(~B)&(C)&(~D);
assign x4=(~A)&(B)&(C)&(D);
assign x5=(B)&(~C)&(D);
assign F1=x1|x2|x3|x4|x5;
endmodule


3)
module EX03(a,b,sum,carry,D,Bo); 
input a,b; 
output sum,carry,D,Bo;
xor g1(sum,a,b);
and g2(carry,a,b); 
wire abar; 
not g3(abar,a); 
xor g4(D,a,b); 
and g5(Bo,abar,b); 
endmodule


4)
## Full_adder
module fulladd_top(a,b,cin,sum,carry);
input a,b,cin;
output sum,carry;
wire w1,w2,w3,w4;       
xor(w1,a,b);
xor(sum,w1,cin);        
and(w2,a,b);
and(w3,b,cin);
and(w4,cin,a);
or(carry,w2,w3,w4);
endmodule 
## Full_subtractor
module fullsub_top(a,b,Bin,BO,DIFF);
input a,b,Bin;
output BO,DIFF;
assign DIFF = a ^ b ^ Bin;
  assign BO = (a & b) | ((a ^ b) & Bin);
endmodule

5)
input d0,d1,d2,d3,d4,d5,d6,d7;
output a0,a1,a2;
assign a0=d1|d3|d5|d7;
assign a1=d2|d3|d6|d7;
assign a2=d4|d5|d6|d7;
endmodule


6)
module sr_flipflop(q, q_bar, s, r, clk, reset);
  input s, r, clk, reset;
  output reg q;
  output q_bar;
  always @(posedge clk) begin
    if (!reset) 
      q <= 1'b0;
    else begin
      case ({s, r})
        2'b01: q <= 1'b0;
        2'b10: q <= 1'b1;
        2'b11: q <= 1'bx;
        default: q <= q;
      endcase
    end
  end
  assign q_bar = ~q;
endmodule


7)
module JK(q, qb,j,k,clock,reset);
  input j,k,clock,reset;
 output reg q, qb;	 
always @ (posedge (clock))
  begin 
   if (!reset)
 begin
 q <= q;
 qb <=qb;
 end       
else
begin
if(j==0&&k==0)
begin
q<=q;
qb<=qb;
end
else if(j!=k)
begin
q<=j;
qb<=k;
end
else if(j==1&&k==1)
begin
q<=~q;
qb<=~qb;
end
end
end       
endmodule


8)
module D_FF(D,Clock,reset,Q);
input D,Clock,reset;
output reg Q;
always@(negedge Clock)//use negative edge clock for triggering condition
if(!reset)
Q<=0;
else
Q<=D;
endmodule


9)
module T_FLIPFLOP( input clk, rst_n, input t,
output reg q,
output q_bar
);
always@(posedge clk) 
begin 
if(!rst_n)
 q<=0;
 else
 if(t)
 q<=~q;
 else
 q<=q;
 end
assign q_bar = ~q;
endmodule

10)
module EXP10(clk, sin, q);
input clk;
input sin;
output [3:0] q;
reg [3:0] q;
always @(posedge clk)
begin
q[0] <= sin;
q[1] <= q[0];
q[2] <= q[1];
q[3] <= q[2];
end
endmodule

11)
module ex11(out,clk,rstn);
input clk,rstn;
output reg [3:0]out;
always @ (posedge clk)
begin
   if(!rstn)
     out<=0;
   else 
     out <= out+1;
end
endmodule
```
