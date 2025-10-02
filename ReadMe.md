# Overzicht vooruitgang

## 26/09
Lezen boek Sequential Decision Analytics and modeling - Warren B. Powell.  
Boek legt een conventie uit om een model op te bouwen voor een sequentieel beslissingsprobleem. 

## 27/09
### Price forecasting   
Hello world example in [deze commit](https://github.com/DeLany123/Thesis/commit/340c0341b57fafdc6a76f11dc324c3bce96f8e41)  
Ik las in het boek dat er ook forecasts gebruikt worden in de modellen, om wat af te wisselen ben ik de haalbaarheid ook een beetje aan het onderzoeken.  
    
![Mijn data-analyse grafiek](plots/model1_prediction_output_sample.png "Voorspelling vs. Werkelijkheid voor 2024")    
**RMSE: 90.77 €/MWh**

Op de grafiek vallen twee belangrijke kenmerken van het model op:

*   **Vertraagde Voorspelling (Lag):** De voorspelde lijn volgt de werkelijke prijs met een kleine vertraging. Het model heeft geleerd dat de vorige prijs de beste gok is voor de volgende.

*   **Afgevlakte Pieken (Smoothing):** De voorspelling is veel vlakker en mist de extreme pieken en dalen. Het model is getraind om grote fouten te vermijden; een wilde gok op een piek is riskant. Daarom kiest het model voor een veiligere, meer gemiddelde voorspelling om de totale fout zo laag mogelijk te houden.  

### Model
**Doel:**  
*Begrijpen waarom en hoe een model omgaat met onzekerheden en waarom dit niet kan vervangen worden door de forecaster.*

De forecaster geeft één "beste" gok voor de toekomst. Het model uit het boek gebruikt een kansverdeling om meerdere mogelijke toekomstscenario's te simuleren. Hierdoor kan een agent leren om een beslissing te nemen die gemiddeld genomen het beste presteert over al die mogelijke uitkomsten, in plaats van te optimaliseren voor maar één voorspelde toekomst.