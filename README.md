```python
1. IMPLEMENTATION OF SYMBOL TABLE
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>  
#define MAX_EXPRESSION_SIZE 100
int main()
 {
    int i = 0, j = 0, x = 0, n, flag = 0;
    void *add[15];   
    char b[MAX_EXPRESSION_SIZE], d[15], c, srch;
    printf("Enter the Expression terminated by $: ");
    while ((c = getchar()) != '$' && i < MAX_EXPRESSION_SIZE - 1)
    {
        b[i++] = c;
    }
    b[i] = '\0';  
    n = i;
    printf("Given Expression: %s\n", b);
    printf("\nSymbol Table\n");
    printf("Symbol\taddr\ttype\n");
    for (j = 0; j < n; j++)
    {
        c = b[j];
        if (isalpha((unsigned char)c))
        { 
            void *p = malloc(sizeof(char));  
            add[x] = p;
            d[x] = c;
            printf("%c\t%p\tidentifier\n", c, p);  
            x++;
        }
    }
    getchar();  
    printf("\nThe symbol to be searched: ");
    srch = getchar();
    for (i = 0; i < x; i++)
    {
        if (srch == d[i])
        {
            printf("Symbol Found\n");
            printf("%c@address %p\n", srch, add[i]);
            flag = 1;
            break;
        }
    }
    if (flag == 0)
    {
        printf("Symbol Not Found\n");
    }
    for (i = 0; i < x; i++)
    {
        free(add[i]);
    }
    return 0;
}



2.GENERATION OF LEXICAL TOKENS LEX/FLEX TOOL
#include <stdio.h>
#include <ctype.h>
#include <string.h>
int isKeyword(char buffer[])
{
    char keywords[5][10] = {"if", "else", "while", "for", "int"};
    for (int i = 0; i < 5; ++i) 
    {
        if (strcmp(buffer, keywords[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

int main() 
{
    char ch, buffer[15];
    char operators[] = "+-*/=";
    int i = 0;
    printf("Enter your input: ");
    while ((ch = getchar()) != EOF) 
    {
        if (strchr(operators, ch)) 
        {
            printf("Operator: %c\n", ch);
        } 
        else if (isalnum(ch)) 
        {
            buffer[i++] = ch;
        } 
        else if ((ch == ' ' || ch == '\n' || ch == '\t') && i != 0) 
        {
            buffer[i] = '\0';
            if (isKeyword(buffer))
            {
                printf("Keyword: %s\n", buffer);
            } 
            else if (isdigit(buffer[0]))
            {
                printf("Number: %s\n", buffer);
            } 
            else
            {
                printf("Identifier: %s\n", buffer);
            }
            i = 0;
        }
    }
    return 0;
}

3. RECOGNITION OF A VALID ARITHMETIC EXPRESSION THAT USES
#include <stdio.h>
#include <ctype.h>
#include <string.h>
int isKeyword(char buffer[])
{
    char keywords[5][10] = {"if", "else", "while", "for", "int"};
    for (int i = 0; i < 5; ++i) 
    {
        if (strcmp(buffer, keywords[i]) == 0) 
        {
            return 1;
        }
    }
    return 0;
}
int main() 
{
    char ch, buffer[15];
    char operators[] = "+-*/=";
    int i = 0;
    printf("Enter your input: ");
    while ((ch = getchar()) != EOF)
    {
        if (strchr(operators, ch)) 
        {
            printf("Operator: %c\n", ch);
        } 
        else if (isalnum(ch)) 
        {
            buffer[i++] = ch;
        } 
        else if ((ch == ' ' || ch == '\n' || ch == '\t') && i != 0)
        {
            buffer[i] = '\0';
            if (isKeyword(buffer))
            {
                printf("Keyword: %s\n", buffer);
            } 
            else if (isdigit(buffer[0])) 
            {
                printf("Number: %s\n", buffer);
            } 
            else 
            {
                printf("Identifier: %s\n", buffer);
            }
            i = 0;
        }
    }

    return 0;
}


4. RECOGNITION OF A VALID VARIABLE WHICH STARTS WITH A LETTER FOLLOWED BY ANY NUMBER OF LETTERS OR DIGITS USING YACC
#include <stdio.h>
#include <string.h>
#include <ctype.h>
void classify_identifiers(char *line) 
{
    char *token = strtok(line, " ,;");
    int found_type = 0;
    while (token != NULL) 
    {
        if (strcmp(token, "int") == 0)
        {
            found_type = 1;
        }
        else if (found_type) 
        {
            if (isalpha(token[0]))
            {
                printf("Identifier is %s\n", token);
            }
        }
        token = strtok(NULL, " ,;");
    }
}

int main()
{
    char line[] = "int a,d;";
    classify_identifiers(line);
    return 0;
}

5.RECOGNITION OF THE GRAMMAR
#include <stdio.h>
#include <string.h>
int is_valid_string(const char *str) 
{
    int len = strlen(str);
    if (str[len - 1] != 'b') 
    {
        return 0;
    }
    for (int i = 0; i < len - 1; i++) 
    {
        if (str[i] != 'a')
        {
            return 0;
        }
    }
    return 1; 
}
int main() 
{
    char str[100];
    printf("Enter a string: ");
    scanf("%s", str);
    if (is_valid_string(str))
    {
        printf("Valid string\n");
    } 
    else 
    {
        printf("Invalid string\n");
    }
    return 0;
}

6.IMPLEMENTATION OF THE BACK END OF THE COMPILER
#include <stdio.h>
#include <string.h>
typedef struct
{
    char opcode[10]; 
    char arg1[10];   
    char arg2[10];   
    char result[10]; 
} Instruction;
void generate_code(Instruction ir[], int count)
{
    printf("Generated Assembly:\n");
    for (int i = 0; i < count; i++) 
    {
        if (strcmp(ir[i].opcode, "ADD") == 0)
        {
            printf("    ADD %s, %s ; Store in %s\n", ir[i].arg1, ir[i].arg2, ir[i].result);
        } 
        else if (strcmp(ir[i].opcode, "SUB") == 0) 
        {
            printf("    SUB %s, %s ; Store in %s\n", ir[i].arg1, ir[i].arg2, ir[i].result);
        } 
        else if (strcmp(ir[i].opcode, "MOV") == 0) 
        {
            printf("    MOV %s, %s\n", ir[i].arg1, ir[i].result);
        } 
        else
        {
            printf("Unknown operation: %s\n", ir[i].opcode);
        }
    }
}
int main() 
{
    Instruction ir[] = {
        {"MOV", "R1", "", "A"},    
        {"ADD", "A", "B", "R2"},   
        {"SUB", "R2", "C", "R3"},  
        {"MOV", "R3", "", "D"}     
    };
    int ir_count = sizeof(ir) / sizeof(Instruction);
    generate_code(ir, ir_count);
    return 0;
}
```




























