@0 //A=0 M=RAM[0]
D=M // D = RAM[0]

@2//A=2 M=RAM[2]
M=D // RAM[2] = RAM[0]  (temp)

@1 //A=1 M=RAM[1]
D=M // D = RAM[1]

@0 //A=0 M=RAM[0]
M=D // RAM[0] = RAM[1]

@2//A=2 M=RAM[2]
D=M // D = old RAM[0]

@1//A=1 M=RAM[1]
M=D // RAM[1] = old RAM[0]
