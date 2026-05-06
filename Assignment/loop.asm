//INITIALIZATION
@0            // A = 0
M=0           // RAM[0] = i = 0 (initialize loop counter)
//LOOP START
(LOOP)        // label LOOP
@0            // A = 0
D=M           // D = i
@1            // A = 1
D=D-M         // D = i - n
@END          // A = address of END
D;JGE         // if i >= n, jump to END (exit loop)
//arr[i] = -1
@10           // A = 10 (base address of array)
D=A           // D = 10
@0            // A = 0
A=D+M         // A = 10 + i (address of arr[i])
M=-1          // arr[i] = -1
//i = i + 1
@0            // A = 0
M=M+1         // i = i + 1
@LOOP         // A = address of LOOP
0;JMP         // jump back to LOOP
//PROGRAM END
(END)         // label END
@END          // A = END
0;JMP         // infinite loop (program finished)