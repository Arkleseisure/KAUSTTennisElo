import pandas as pd  # Importing pandas for data manipulation
import math  # Importing math for mathematical operations
import os  # Importing os for file handling
import random

'''
ks = [60] #[30, 60, 90, 120, 150]
prelim_ks = [120] #[60, 120, 180, 240, 300]
for initial_k in ks:
    for preliminary_k in prelim_ks:
        print('initial k:', initial_k)
        print('preliminary k:', preliminary_k)
'''
# Constants for Elo rating adjustments
initial_k = 60  # Base K-factor for regular Elo updates
preliminary_k = 120  # Higher base K-factor for players with few games played
starting_matches = 5  # Number of games before a player is considered for regular K-factor
# A player's first elo goes off their score in their first game, if one player scores 0,
# this would give an infinite rating gap, so the loser is given a point.
zero_score = 1
points_per_game = 6.5  # number of points won in an average game (to define relative intensity of tiebreaks and sets)
games_per_intensity_unit = 2  # we're using one intensity unit as being one service game per player.
games_vs_preliminary_weight = 0.5  # how much do games against other preliminary players count
# towards the number of starting matches



# Function to calculate the expected score based on two players' Elo ratings
def expected_score(elos):
    return 1 / (1 + math.pow(10, (elos[0] - elos[1]) / 400))


# Function to calculate the match intensity factor based on the match score
def calculate_match_intensity(scores, match_type):
    # Calculate the total number of games played
    total_games = scores[0] + scores[1]

    # Adjust the scaling for matches that are just tiebreaks
    if match_type == 'points':
        total_games /= points_per_game

    # K-factor proportional to the length of the match
    return total_games / games_per_intensity_unit


def get_game_score_from_points_score(winner_score):
    # formula for probability of winning a game of tennis given the probability of winning a point
    winner_score = (winner_score ** 4) * (1 + 4 * (1 - winner_score) + 10 * ((1 - winner_score) ** 2) +
                                          20 * winner_score * ((1 - winner_score) ** 3)
                                          / (1 - 2 * winner_score * (1 - winner_score)))
    return winner_score, 1 - winner_score


# Function to update Elo ratings after a match
def update_elo(elos, scores, k, type):
    loser_expected = expected_score(elos)  # Expected score for the losing player
    winner_expected = 1 - loser_expected  # Expected score for the winning player

    total_score = sum(scores)  # Total score of the game
    winner_score = scores[0] / total_score
    loser_score = scores[1] / total_score

    if type == 'points':
        winner_score, loser_score = get_game_score_from_points_score(winner_score)

    # Update Elo ratings for both players
    new_winner_elo = elos[0] + k * (winner_score - winner_expected)
    new_loser_elo = elos[1] + k * (loser_score - loser_expected)

    mse = (winner_score - winner_expected) ** 2
    random_mse = (random.random() - winner_score) ** 2
    return new_winner_elo, new_loser_elo, mse, random_mse  # Return the updated Elo ratings and the squared error


# Function to find a player's Elo rating given opponent's Elo and scores
def find_elo(opp_elo, opp_score, own_score):
    if own_score == 0:
        own_score = zero_score  # Use fallback score if player's score is zero
    elif opp_score == 0:
        opp_score = zero_score
    score = own_score / (own_score + opp_score)  # Calculate the proportionate score
    # Calculate and return the Elo rating
    return round(opp_elo - 400 * math.log10(1 / score - 1), 1)


# Function to load a dataframe from an Excel file with error handling
def load_dataframe(file_path, columns=None):
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        if columns:
            for column in columns:
                if column not in df.columns:
                    raise ValueError(f"Missing expected column: {column}")
        return df
    else:
        return pd.DataFrame(columns=columns)  # Return an empty dataframe if the file does not exist


# Function to save historical results back to the text file
def save_historical_results(file_path, results):
    with open(file_path, 'w') as f:
        for item in results:
            f.write(item)


# Load current and preliminary Elo scores
elo_scores = load_dataframe('elo_scores.xlsx', columns=['player', 'elo'])
prelim_elo_scores = load_dataframe('preliminary_elos.xlsx', columns=['player', 'elo', 'games'])

# Load the game results
games = load_dataframe('game_results.xlsx', columns=['winner', 'loser', 'win_score', 'lose_score', 'type'])

# Load historical results
historical_results_path = 'historical_results.txt'
if os.path.exists(historical_results_path):
    with open(historical_results_path, 'r') as f:
        prev_results = f.readlines()
else:
    prev_results = []

# correct predictions in the form [total, full (both have full ratings),
#                                  half (one has a full rating, one preliminary), no (both preliminary ratings)]
num_correct = [0, 0, 0, 0]
total = [0, 0, 0, 0]
mse = [0, 0, 0, 0]
mae = [0, 0, 0, 0]
random_correct = [0, 0, 0, 0]
random_mse = [0, 0, 0, 0]

# Process each game in the game results
for index, game in games.iterrows():
    players = [game['winner'], game['loser']]  # Extract the players from the game data
    scores = [game['win_score'], game['lose_score']]  # Extract the scores from the game data
    match_type = game['type']  # Get the match type (e.g., 'games' or 'points')
    match_date = game['date']
    is_preliminary = [False, False]  # Initialize flags for preliminary players
    elos = [0, 0]  # Initialize Elo ratings for the players

    for i in range(2):
        if players[i] in elo_scores['player'].values:
            elos[i] = elo_scores.loc[elo_scores['player'] == players[i], 'elo'].values[0]
        elif players[i] in prelim_elo_scores['player'].values:
            is_preliminary[i] = True
            elos[i] = prelim_elo_scores.loc[prelim_elo_scores['player'] == players[i], 'elo'].values[0]
        else:
            # New player, assign default Elo rating
            elos[i] = 1500
            is_preliminary[i] = True

    index = 0
    if is_preliminary[0] != is_preliminary[1]:
        index = 2
    elif is_preliminary[0]:
        index = 3
    else:
        index = 1

    total[index] += 1
    total[0] += 1
    correct = False
    if elos[0] >= elos[1]:
        correct = True
        num_correct[0] += 1
        num_correct[index] += 1

    if random.randint(0, 1) == 0:
        random_correct[0] += 1
        random_correct[index] += 1

    # Calculate the match intensity factor based on the length of the match and its type
    intensity_factor = calculate_match_intensity(scores, match_type)

    # Determine the K-factor adjusted by the match intensity
    k_factor = preliminary_k if any(is_preliminary) else initial_k
    k_factor_adjusted = k_factor * intensity_factor

    # Update Elo ratings based on the game result
    new_winner_elo, new_loser_elo, squared_error, random_squared_error = \
        update_elo(elos, scores, k_factor_adjusted, match_type)
    new_elos = [round(new_winner_elo, 1), round(new_loser_elo, 1)]  # Round the new Elo ratings

    mse[0] += squared_error
    mse[index] += squared_error
    mae[0] += math.sqrt(squared_error)
    mae[index] += math.sqrt(squared_error)
    random_mse[0] += random_squared_error
    random_mse[index] += random_squared_error

    # Update the Elo scores in the appropriate dataframes
    for i in range(2):
        if is_preliminary[i]:  # Handle updates for the preliminary player
            game_weight = 1
            if is_preliminary[1 - i]:
                game_weight = games_vs_preliminary_weight

            matches_played = prelim_elo_scores.loc[prelim_elo_scores['player'] == players[i], 'games'].values[0] \
                if players[i] in prelim_elo_scores['player'].values else 0
            if matches_played == 0:  # If this is the player's first game
                new_elos[i] = find_elo(elos[1 - i], scores[1 - i], scores[i])
                new_row = pd.DataFrame({'player': [players[i]], 'elo': [new_elos[i]], 'games': [game_weight]})
                prelim_elo_scores = pd.concat([prelim_elo_scores, new_row], ignore_index=True)
            elif matches_played >= starting_matches - 1:  # If player has finished preliminary games
                new_row = pd.DataFrame({'player': [players[i]], 'elo': [new_elos[i]]})
                elo_scores = pd.concat([elo_scores, new_row], ignore_index=True)
                prelim_elo_scores = prelim_elo_scores[prelim_elo_scores['player'] != players[i]]
            else:  # Update the preliminary player's Elo and games count
                prelim_elo_scores.loc[prelim_elo_scores['player'] == players[i], 'elo'] = new_elos[i]
                prelim_elo_scores.loc[prelim_elo_scores['player'] == players[i], 'games'] = matches_played + game_weight
        elif not (is_preliminary[0] or is_preliminary[1]):  # updates the scores if both players aren't preliminary
            elo_scores.loc[elo_scores['player'] == players[i], 'elo'] = new_elos[i]
        else:  # rating for an established player doesn't change against a preliminary player
            new_elos[i] = elos[i]

    # Append the results to the historical results log
    prev_results.append(f' {match_date}: ({round(new_elos[0] - elos[0], 1)}) ({new_elos[0]}) {players[0]} {scores[0]}-{scores[1]} '
                        f'{players[1]} ({new_elos[1]}) ({round(new_elos[1] - elos[1], 1)})\n')

# Save the updated Elo scores back to the files
elo_scores.sort_values(by="elo", ascending=False, inplace=True)
elo_scores.to_excel('updated_elo_scores.xlsx', index=False)
elo_scores.to_csv('updated_elo_scores.csv', index=False)
prelim_elo_scores.sort_values(by="elo", ascending=False, inplace=True)
prelim_elo_scores.to_excel('updated_prelim_elo_scores.xlsx', index=False)

# Save the updated historical results back to the text file
save_historical_results(historical_results_path, prev_results)

prediction_types = ['total', 'full', 'half', 'no']
print(f"Elo scores updated successfully! Processed {len(games)} games.")
for i in range(len(prediction_types)):
    print(prediction_types[i] + ' predictions:')
    print(f'Algorithm worked at {round(100 * num_correct[i]/total[i], 1)}% accuracy over {total[i]} games.')
    print(f'Mean squared error was {round(mse[i]/total[i], 3)} and the sqrt of that is '
          f'{round(math.sqrt(mse[i]/total[i]), 3)}')
    print(f'Mean absolute error was {round(mae[i]/total[i], 3)}')
    print(f'Random worked at {round(100 * random_correct[i]/total[i], 1)}% accuracy and had a mse of '
          f'{round(random_mse[i]/total[i], 2)}, sqrt {round(math.sqrt(random_mse[i]/total[i]), 2)}')
