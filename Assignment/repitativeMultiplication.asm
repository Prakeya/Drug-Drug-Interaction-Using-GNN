@3
M=0
@i
M=0

(LOOP)
@i
D=M
@2
D=D-M
@END
D;JGE

@1
D=M
@3
M=M+D

@i
M=M+1
@LOOP
0;JMP

(END)
@END
0;JMP