@1      //A=1 M=RAM[1]
D=M    // D = RAM[1]
@2     //A=2 M=RAM[2]
D=D+M   // D = RAM[1] + RAM[2]
@3      //A=3 M=RAM[3]
M=D    // RAM[3] = D
@6
0;JMP
