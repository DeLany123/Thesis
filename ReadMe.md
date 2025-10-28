# Codebase
The relevant code files are under the `Simulation/suite_simple_trading/` directory.  
The starting point is the `driver.py` script, which handles command-line arguments and orchestrates the training, evaluation, and tuning of models.

# Run the Project

1.  **Set up the Environment:**
    First, create and activate a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    Install all the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train, Evaluate, and Tune Models:**
    Navigate to the project's root directory (`ThesisGit`) and execute the driver script as a module. You must specify a command (`rl` or `heuristic`) followed by its arguments. New plots are saved to the `plots` directory.

    *   **Train the RL Agent:**
        ```bash
        python -m Simulation.suite_simple_trading.driver rl --mode train
        ```

    *   **Evaluate the trained RL Agent:**
        ```bash
        python -m Simulation.suite_simple_trading.driver rl --mode run
        ```
    
    *   **Evaluate the Heuristic Policy:**
        ```bash
        python -m Simulation.suite_simple_trading.driver heuristic --mode run
        ```

    *   **Run a Grid Search** for the heuristic policy:
        ```bash
        python -m Simulation.suite_simple_trading.driver heuristic --mode gridsearch
        ```
    
    *   **Run Cross-Validation** for the RL agent:
        ```bash
        python -m Simulation.suite_simple_trading.driver rl --mode test
        ```

### Command-Line Arguments

The script uses two main commands: `rl` and `heuristic`. Each has its own set of options.

#### `rl` Command
Used for training, evaluating, and testing the Reinforcement Learning agent.

*   `--mode`: (Optional) Specifies the operating mode for the RL agent. Defaults to `run`.
    *   `train`: Trains the DQN agent on the training dataset and saves the model.
    *   `run`: Evaluates the pre-trained DQN agent on the test dataset.
    *   `test`: Performs a full time-series cross-validation.
*   `--start-date`: (Optional, for `run` mode) Sets the start of the plot window. Format: `"YYYY-MM-DD"`.
*   `--end-date`: (Optional, for `run` mode) Sets the end of the plot window. Format: `"YYYY-MM-DD"`.

#### `heuristic` Command
Used for evaluating and tuning the rule-based heuristic policy.

*   `--mode`: (Optional) Specifies the operating mode for the heuristic. Defaults to `run`.
    *   `run`: Executes a single simulation run on the test dataset.
    *   `gridsearch`: Performs a full grid search to find optimal `buy` and `sell` thresholds.
*   `--buy`: (Optional, for `run` mode) Sets the buying/charging price threshold. Defaults to `10.0`.
*   `--sell`: (Optional, for `run` mode) Sets the selling/discharging price threshold. Defaults to `120.0`.
*   `--start-date`: (Optional, for `run` mode) Sets the start of the plot window. Format: `"YYYY-MM-DD"`.
*   `--end-date`: (Optional, for `run` mode) Sets the end of the plot window. Format: `"YYYY-MM-DD"`.

### Examples

*   **Run the RL agent and plot the first week of the test period:**
    ```bash
    python -m Simulation.suite_simple_trading.driver rl --mode run --start-date "2025-01-01" --end-date "2025-01-08"
    ```

*   **Run the heuristic with custom thresholds:**
    ```bash
    python -m Simulation.suite_simple_trading.driver heuristic --mode run --buy 0 --sell 150
    ```
    

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

## 2/10
### Model
Afwerken van eerste iteratie model alsook het neerschrijven ervan in een LaTeX document. Zie model.pdf .

**Code implementatie**
Eerste simulatie op heuristiek policy in Simulation/first.py

## 6/10
### Simulatie
Eerste simulatie op heuristiek policy "buy low, sell high" in Simulation/first.py [deze commit](https://github.com/DeLany123/Thesis/commit/d86b5aeaedef0a6e1726759b267f536b76d8d68f)
Deze simulatie neemt als input de grote en charge rate van een batterij, alsook een range van mogelijke parameterwaarden 
voor de heuristiek.

De simulatie gebruikt de volgende input (nu met testdata ingevuld):
- **Batterij Capaciteit:** 10.0 MWh
- **Laad-/Ontlaadvermogen:** 2.0 MW
- **Koopdrempels (testbereik):** Van 80 tot 100 EUR/MWh
- **Verkoopdrempels (testbereik):** Van 105 tot 125 EUR/MWh

Met als resultaat
```bash
--- Grid Search Complete ---
Optimal Buy Threshold: 90.00 EUR/MWh
Optimal Sell Threshold: 105.00 EUR/MWh
Resulting Best Profit: 300812.77 EUR
```

Achter de simulatie kwam het besef dat ik nu enkel een "trader" die enkel met een Battery Energy Storage System (BES) kijkt
naar de grootste winst. Er wordt nog **geen rekening gehouden met de demand van de gebruiker.**

De demand van de gebruiker moet vervuld worden door ofwel de batterij te gebruiken, ofwel op de imbalance markt te kopen.
Het kan zelfs zijn dat deze laatste voordeliger lijkt dan de batterij te gebruiken aangezien op een later tijdstip de
elektriciteit kan verkocht worden aan een veel hogere prijs.

Met deze nieuwe inzichten kon ik het model terug wat aanpassen.

### Model
Meer decision variables alsook meer constraints toegevoegd aan het model. Zie model.pdf .
De rest moet nog gewijzigd worden aan het model. Zie nieuw document Thesis_Model_1.pdf .

## 10/10
### Simulatie
Infrastruur verder uitgewerkt zodat meerdere soorten policies en netwerken op hetzelfde model getest kunnen worden.  
Hierbij ook de eerste simulatie afgewerkt en een plotting functie toegevoegd waarbij als parameter de startdatum en lengte
meegegeven kan worden. Zie Simulation/suite_simple_trading/plotting.py [deze commit](https://github.com/DeLany123/Thesis/commit/2a5a2326e22d02cdcd2c4da1de3d3cae7d09cae4)
![simulation_plot](plots/model1_easy_heuristic_simulation.png "Voorspelling vs. Werkelijkheid voor 2024")    
In de plot zien we de beslissing van de batterij om op te laden en de prijs van de imbalanceprijs.

### Eerste meeting Thesis
Overleg over het basic model en alle mogelijke policy families. Alsook de reward functie is besproken.
Besproken om te onderzoeken.
<ol>
<li>Betere reward functie, reward is pas aan het einde van het kwartier.</li>
<li>Policyfamilies bekijken, uittesten, welke zien er veelbelovend uit, welke niet.</li>
<li>Invloed van forecasting op het model</li>
</ol>

## 17/10
### Model
DQN agent opgezet in Simulation/suite_simple_trading/agents/dqn_agent.py.
State space uitgebreid met meerdere relative oplaad levels.

### Resulaten
![DQN Training Progress](plots/simulation_results_dqn_1.png "DQN Training Progress")
We zien dat het model enkel wilt kopen als het een negatief prijs heeft gezien in de state. Hierna probeert het heel 
snel terug te verkopen voor een positieve prijs. En houdt waarschijnlijk nog geen rekening met een hoge prijs in de toekomst.

## 28/10
Lang gewerkt aan revampen van het model. Meer in de literatuur gedoken om bij te lezen over de verschillende oplosmethodes.
Geleerd hoe DQN werkt.

### Overleg
Overleg gehad met Professor Claeys en Naim over aanpak van het verbeteren van de resultaten.
Conclusie was het niet uitbereiden van het model, dit zou de resultaten nog oninterpreteerbaarder maken.
De volgende stap is om Decision masking te implementeren. Nu worden sommige acties geblockt omdat bijvoorbeeld de batterij vol is.
Het netwerk leert wel nog dat deze beslissing de beste is, en slaat deze op in de lookup table (zie DQN). We moeten er nu voor
zorgen dat deze acties niet meer gekozen kunnen worden in het netwerk zelf. 
In de volgende repo zouden er modellen moeten zijn die decision masking ondersteunen.  
https://sb3-contrib.readthedocs.io/en/master/  

Daarnaast kan er ook gekeken worden of er niet aan aggregatie kan gedaan worden van actions. Dat bijvoorbeeld elke 5 minuten
een actie genomen wordt voor de volgende 5. Hiermee kunnen we effecten van de delayed reward wat verminderen.