8 a)


``` 
#include "main.h"

TIM_HandleTypeDef htim;

void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM_Init(void);

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_TIM_Init();

  HAL_TIM_BASE_Start(&htim);
  HAL_TIM_PWM_Init(&htim);
  HAL_TIM_PWM_Start(&htim, TIM_CHANNEL_1);

  while (1)
  {
    // Your application code here
  }
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  // Configure the system clock
  // ...

  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_GPIO_Init(void)
{
  // Configure GPIOC pin 7 as PWM output
  __HAL_RCC_GPIOC_CLK_ENABLE();
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Pin = GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Alternate = GPIO_AF1_TIM3;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}

static void MX_TIM_Init(void)
{
  // Configure Timer for PWM
  __HAL_RCC_TIM3_CLK_ENABLE();
  htim.Instance = TIM3;
  htim.Init.Prescaler = 0;
  htim.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim.Init.Period = 1000; // PWM Period
  htim.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_PWM_Init(&htim) != HAL_OK)
  {
    Error_Handler();
  }

  TIM_OC_InitTypeDef sConfigOC = {0};
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 750; // 75% duty cycle
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
}
```
