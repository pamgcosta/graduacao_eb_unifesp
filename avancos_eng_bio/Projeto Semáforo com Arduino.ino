const int sema1_verde=8;
const int sema1_amarelo=9;
const int sema1_vermelho=10;

const int sema2_verde=5;
const int sema2_amarelo=6;
const int sema2_vermelho=7;

const int res_timer=700; 

int conta_timer=0;
int espera=0;
int ciclo=0;


void setup() {
  pinMode(13, OUTPUT);          
  pinMode(sema1_verde, OUTPUT);
  pinMode(sema1_amarelo, OUTPUT);
  pinMode(sema1_vermelho, OUTPUT);
  pinMode(sema2_verde, OUTPUT);
  pinMode(sema2_amarelo, OUTPUT);
  pinMode(sema2_vermelho, OUTPUT);
}

void loop() {
  switch(ciclo) {
    case 0:   
      digitalWrite(sema1_verde, HIGH);
      digitalWrite(sema1_vermelho, LOW);
      digitalWrite(sema2_vermelho, HIGH);
      digitalWrite(sema2_amarelo, LOW);
      espera=4;
      break;
    case 1:   
      digitalWrite(sema1_amarelo, HIGH);
      digitalWrite(sema1_verde, LOW);
      espera=2;
      break;
    case 2:   
      digitalWrite(sema1_vermelho, HIGH);
      digitalWrite(sema1_amarelo, LOW);
      digitalWrite(sema2_verde, HIGH);
      digitalWrite(sema2_vermelho, LOW);
      espera=4;
      break;
    case 3:   
      digitalWrite(sema2_amarelo, HIGH);
      digitalWrite(sema2_verde, LOW);
      espera=2;
      break;
  }    
  digitalWrite(13, HIGH);
  delay(res_timer/2);
  digitalWrite(13, LOW);
  conta_timer++;
  if (conta_timer>=espera) {
    conta_timer=0;
    ciclo++;
    if (ciclo>4) {
      ciclo=0;
    }  
  }  
  delay(res_timer/2);
}