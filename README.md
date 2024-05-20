 ```python
1. arithmetic operations

org 100h
mov AH,0A2H;
mov BH,0B3H;
add AH,BH;
mov [1554H],AH;
ret

org 100h
mov AH,0C4H;
mov BH,0B2H;
sub AH,BH;
mov [1424h],AH;
ret

org 100h
mov AL,0C4H;
mov BL,0B2H;
mul BL;
mov [1334H],AX;
ret

org 100h
mov AL,0D5H;
mov BL,0A2H;
div BL;
mov [1364H],AX;
ret

2. logical operations

org 100h
MOV bx,1000h;
AND bx,1111h;
MOV [0040h+02],bx;
ret

org 100h
MOV ax,[0070h];
MOV bx,1000h;
OR ax,bx;
MOV [0060h],ax;
ret

org 100h
MOV bx,0060h;
MOV ax,[bx]; 
NOT al;
MOV [0060h+04],ax;
ret

org 100h
MOV bx,0050h;
MOV ax,[bx]; 
XOR ax,bx;
MOV [0050h+03],ax;
ret


3. blink the led 5 times with a delay of 500 ms

21.#include<stdbool.h>
bool button_status;
void push_button();
95.void push_button()
  {button_status = HAL_GPIO_ReadPin(GPIOC,GPIO_PIN_13);
	if(button_status==0)
	{
		HAL_GPIO_WritePin(GPIOA,GPIO_PIN_5,GPIO_PIN_SET);
		HAL_Delay(500);
	}
	else
	{
		HAL_GPIO_WritePin(GPIOA,GPIO_PIN_5,GPIO_PIN_RESET);
		HAL_Delay(500);
	}
}
  for(int i=0;i<=5;i++)
  {
	  push_button();
  }


4. pushbutton

21.#include<stdbool.h>
void push_button();
109.void push_button()
{
	button_status = HAL_GPIO_ReadPin(GPIOC,GPIO_PIN_13);
	if(button_status==0)
	{
		HAL_GPIO_WritePin(GPIOA,GPIO_PIN_5,GPIO_PIN_SET);
		HAL_Delay(500);
		HAL_GPIO_WritePin(GPIOA,GPIO_PIN_5,GPIO_PIN_RESET);
				HAL_Delay(500);
	}
	else
	{
		HAL_GPIO_WritePin(GPIOA,GPIO_PIN_5,GPIO_PIN_RESET);
		HAL_Delay(500);
	}
}

5. name/roll

Lcd_PortType ports[] = { GPIOA, GPIOA, GPIOA, GPIOA };
     Lcd_PinType pins[] = {GPIO_PIN_3, GPIO_PIN_2, GPIO_PIN_1, GPIO_PIN_0};
     Lcd_HandleTypeDef lcd;
     lcd = Lcd_create(ports, pins, GPIOB, GPIO_PIN_0, GPIOB, GPIO_PIN_1, LCD_4_BIT_MODE);
     Lcd_cursor(&lcd, 0,0);
     Lcd_string(&lcd, "Leann/212222230074");

6. calc

Lcd_PortType ports[] = {GPIOA,GPIOA,GPIOA,GPIOA};
		Lcd_PinType pins[] ={GPIO_PIN_3,GPIO_PIN_2,GPIO_PIN_1,GPIO_PIN_0};
		Lcd_HandleTypeDef lcd;
		lcd = Lcd_create(ports,pins,GPIOB,GPIO_PIN_0,GPIOB,GPIO_PIN_1,LCD_4_BIT_MODE);
			HAL_GPIO_WritePin(GPIOC,GPIO_PIN_0,GPIO_PIN_SET);
				HAL_GPIO_WritePin(GPIOC,GPIO_PIN_1,GPIO_PIN_SET);
				HAL_GPIO_WritePin(GPIOC,GPIO_PIN_2,GPIO_PIN_RESET);
				HAL_GPIO_WritePin(GPIOC,GPIO_PIN_3,GPIO_PIN_SET);

				col1 = HAL_GPIO_ReadPin(GPIOC,GPIO_PIN_4);
				col2 = HAL_GPIO_ReadPin(GPIOC,GPIO_PIN_5);
				col3 = HAL_GPIO_ReadPin(GPIOC,GPIO_PIN_6);
				col4 = HAL_GPIO_ReadPin(GPIOC,GPIO_PIN_7);
				Lcd_cursor(&lcd,0,1);
				if(!col1)
				{
					Lcd_string(&lcd,"key 1\n");
					HAL_Delay(300);
				}
				else if(!col2)
				{
					Lcd_string(&lcd,"key 2\n");
					HAL_Delay(300);
				}
				else if(!col3)
				{
					Lcd_string(&lcd,"key 3\n");
					HAL_Delay(300);
				}
				else if(!col4)
				{
					Lcd_string(&lcd,"key -\n");
					HAL_Delay(300);
				}

	}


7.  timers

91.HAL_TIM_Base_Start(&htim2);
HAL_TIM_PWM_Init(&htim2);
HAL_TIM_PWM_Start(&htim2,TIM_CHANNEL_1);

