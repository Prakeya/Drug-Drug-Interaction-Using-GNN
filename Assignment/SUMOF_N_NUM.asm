@i
M=1

@sum
M=0

(LOOP)
@i
D=M
@sum
M=M+D

@i
M=M+1

@0
D=M
@i
D=D-M

@LOOP
D;JGE

@sum
D=M
@1
M=D

(END)
@END
0;JMP