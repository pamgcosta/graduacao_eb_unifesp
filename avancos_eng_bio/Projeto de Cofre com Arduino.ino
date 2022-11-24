const int display[7] = {2, 3, 4, 5, 6, 7, 8};
const int leds[3] = {9, 10, 11};
const int botao[3] = {12, 13, 14};
const int led_verm = 0, led_verde = 1;

const int senha[3] = {1, 2, 3};

int estado_botao[3];

int cont[3] = {0, 0, 0};

int pos = 0;

void setup(){
	//Set display leds
	pinMode(display[0], OUTPUT);
	pinMode(display[1], OUTPUT);
	pinMode(display[2], OUTPUT);
	pinMode(display[3], OUTPUT);
	pinMode(display[4], OUTPUT);
	pinMode(display[5], OUTPUT);
	pinMode(display[6], OUTPUT);

	//Set leds
	pinMode(leds[0], OUTPUT);
	pinMode(leds[1], OUTPUT);
	pinMode(leds[2], OUTPUT);
	pinMode(led_verm, OUTPUT);
	pinMode(led_verde, OUTPUT);

	//Set botao
	pinMode(botao[0], INPUT); //incrementa
	pinMode(botao[1], INPUT); //decrementa
	pinMode(botao[2], INPUT); //confirma
}

void loop(){
	trata_display(cont[pos]);

	estado_botao[0] = digitalRead(botao[0]);
	estado_botao[1] = digitalRead(botao[1]);
	estado_botao[2] = digitalRead(botao[2]);

	if(estado_botao[0]){
		if(cont < 9){
			cont[pos]++;
		} else{
			cont[pos] = 0;
		}
	}
	
	if(estado_botao[1]){
		if(cont > 0){
			cont[pos]--;
		}	else{
			cont[pos] = 9;
		}
	}

	if(estado_botao[2]){
		digitalWrite(leds[pos], LOW);
		pos++;

		if(pos == 3){
			if(senha[0] == pos[0] && senha[1] == pos[1] && senha[2] == pos[2]){
				digitalWrite(led_verde, HIGH);
			} else{
				digitalWrite(led_verm, HIGH);
			}
		} else{
			digitalWrite(leds[pos], HIGH);
		}
	}
}

void trata_display(int x){
	switch(x){
		case 0:
			digitalWrite(display[0], HIGH);
			digitalWrite(display[1], HIGH);
			digitalWrite(display[2], HIGH);
			digitalWrite(display[3], HIGH);
			digitalWrite(display[4], HIGH);
			digitalWrite(display[5], HIGH);
			digitalWrite(display[6], LOW);
		break;
		
		case 1:
			digitalWrite(display[0], LOW);
			digitalWrite(display[1], HIGH);
			digitalWrite(display[2], HIGH);
			digitalWrite(display[3], LOW);
			digitalWrite(display[4], LOW);
			digitalWrite(display[5], LOW);
			digitalWrite(display[6], LOW);
		break;
		
		case 2:
			digitalWrite(display[0], HIGH);
			digitalWrite(display[1], HIGH);
			digitalWrite(display[2], LOW);
			digitalWrite(display[3], HIGH);
			digitalWrite(display[4], HIGH);
			digitalWrite(display[5], LOW);
			digitalWrite(display[6], HIGH);
		break;
		
		case 3:
			digitalWrite(display[0], HIGH);
			digitalWrite(display[1], HIGH);
			digitalWrite(display[2], HIGH);
			digitalWrite(display[3], HIGH);
			digitalWrite(display[4], LOW);
			digitalWrite(display[5], LOW);
			digitalWrite(display[6], HIGH);
		break;
		
		case 4:
			digitalWrite(display[0], LOW);
			digitalWrite(display[1], HIGH);
			digitalWrite(display[2], HIGH);
			digitalWrite(display[3], HIGH);
			digitalWrite(display[4], LOW);
			digitalWrite(display[5], HIGH);
			digitalWrite(display[6], HIGH);
		break;
		
		case 5:
			digitalWrite(display[0], HIGH);
			digitalWrite(display[1], LOW);
			digitalWrite(display[2], HIGH);
			digitalWrite(display[3], HIGH);
			digitalWrite(display[4], LOW);
			digitalWrite(display[5], HIGH);
			digitalWrite(display[6], HIGH);
		break;
		
		case 6:
			digitalWrite(display[0], HIGH);
			digitalWrite(display[1], LOW);
			digitalWrite(display[2], HIGH);
			digitalWrite(display[3], HIGH);
			digitalWrite(display[4], HIGH);
			digitalWrite(display[5], HIGH);
			digitalWrite(display[6], HIGH);
		break;
	
		case 7:
			digitalWrite(display[0], HIGH);
			digitalWrite(display[1], HIGH);
			digitalWrite(display[2], HIGH);
			digitalWrite(display[3], HIGH);
			digitalWrite(display[4], LOW);
			digitalWrite(display[5], LOW);
			digitalWrite(display[6], LOW);
		break;
		
		case 8:
			digitalWrite(display[0], HIGH);
			digitalWrite(display[1], HIGH);
			digitalWrite(display[2], HIGH);
			digitalWrite(display[3], HIGH);
			digitalWrite(display[4], HIGH);
			digitalWrite(display[5], HIGH);
			digitalWrite(display[6], HIGH);
		break;
		
		case 9:
			digitalWrite(display[0], HIGH);
			digitalWrite(display[1], HIGH);
			digitalWrite(display[2], HIGH);
			digitalWrite(display[3], HIGH);
			digitalWrite(display[4], LOW);
			digitalWrite(display[5], LOW);
			digitalWrite(display[6], HIGH);
		break;
	}
}
