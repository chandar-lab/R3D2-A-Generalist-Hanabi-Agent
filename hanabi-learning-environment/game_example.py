# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code demonstrating the Python Hanabi interface."""

from __future__ import print_function

import numpy as np
import pyhanabi
import re
import sys


def process_cards(cards):
  """
  Process a list of cards and convert them into a human-readable format.

  Args:
      cards (list): List of card codes where each code consists of a color code and a number.

  Returns:
      str: A formatted string representing each card in a human-readable format.
  """
  # Mapping of color codes to their corresponding colors
  color_mapping = {'Y': 'Yellow', 'B': 'Blue', 'R': 'Red', 'W': 'White', 'G': 'Green', 'X': 'Unknown'}

  # Output list to store formatted card information
  output = []

  # Process each card in the input list
  for card in cards:
    # Convert card to string for consistent processing
    card_str = str(card)

    # Extract color and number from the card code
    color_code = card_str[0]
    color = color_mapping.get(color_code, 'Unknown')
    number = card_str[1:]

    # Format and append card information to the output list
    output.append(f"{color} colour card with number {number}")

  # Join the formatted card information into a single string and return
  return ', '.join(output)


def process_fireworks(fireworks):
  """
  Process a list of fireworks and generate a descriptive text.

  Args:
      fireworks (list): List of integers representing the count of different colored fireworks.

  Returns:
      str: A formatted string describing the fireworks display.
  """
  # Define the order of colors in the fireworks list
  color_order = ['Red', 'Yellow', 'Green', 'White', 'Blue']

  # Generate a descriptive text based on the input fireworks list
  processed_firework_text = f"The fireworks display includes {fireworks[0]} {color_order[0]} colour fireworks, " \
                            f"{fireworks[1]} {color_order[1]} colour fireworks, {fireworks[2]} {color_order[2]} colour fireworks, " \
                            f"{fireworks[3]} {color_order[3]} colour fireworks, and {fireworks[4]} {color_order[4]} colour fireworks."

  return processed_firework_text


def extract_knowledge(state_knowledge):
    """
    Extract knowledge information from a string representing the state knowledge.

    Args:
        state_knowledge (str): A string containing knowledge information.

    Returns:
        dict: A dictionary containing extracted knowledge information.
    """
    # Dictionary to store extracted knowledge
    knowledge_dict = {}

    # Convert state_knowledge to string for consistent processing
    state_knowledge = str(state_knowledge)

    # Define a regex pattern to extract information between 'Hands:' and 'Deck size:'
    pattern = re.compile(r'Hands:(.*?)Deck size:', re.DOTALL)

    # Find all matches in the input text
    matches = pattern.findall(state_knowledge)

    # Process each match
    for match in matches:
        knowledge_hand = match.strip()

    # Split the knowledge_hand into lines
    lines = knowledge_hand.strip().split('\n')

    # Counter for tracking different sections
    counter = 0

    # Process each line
    for line in lines:
        # Skip irrelevant lines
        if line == "Cur player":
            continue
        if line == '-----':
            counter += 1
            continue

        # Initialize a list for the current counter if not present in the dictionary
        if counter not in knowledge_dict:
            knowledge_dict[counter] = []

        # Extract relevant information and append to the list
        knowledge_dict[counter].append(line.split(' || ')[1].split('|')[0])

    return knowledge_dict




def get_llm_observation(state, game_parameters):
  """
  Generate a natural language understanding (NLU) observation based on the current state of the Hanabi game.
  SAMPLE FORMAT:
  It is a 3 Player Hanabi game. The current player is 2. There is only 1 life token; when it is 0, it's game over. There are 0 tokens to give a piece of information to other players. The fireworks display includes 0 Red colour fireworks, 0 Yellow colour fireworks, 0 Green colour fireworks, 0 White colour fireworks, and 0 Blue colour fireworks. The deck consists of 33. We can see other player cards except ours , Player 0 has Blue colour card with number 5, Red colour card with number 5, Yellow colour card with number 1, White colour card with number 1, Green colour card with number 1, Player 1 has Yellow colour card with number 1, Green colour card with number 1, Blue colour card with number 4, Green colour card with number 4, Yellow colour card with number 5. The knowledge about our current cards is Unknown colour card with number X, Green colour card with number 5, Unknown colour card with number X, Unknown colour card with number X, Unknown colour card with number X.

  Args:
      state (HanabiState): An object representing the current state of the Hanabi game.

  Returns:
      str: A formatted string providing an observation for a language model.
  """
  # Get information about other players' hands
  other_player_info = state.player_hands()
  final_player_infor_string =""
  for i in range(0, len(other_player_info)):
    if i == state.cur_player():
      continue
    else:
      other_player_info_string = ", Player " + str(i) + " has "

      final_player_infor_string += other_player_info_string +  process_cards(other_player_info[i])

  # Get knowledge about the current player's cards
  knowledge_di = extract_knowledge(state)

  # Generate the observation string
  llm_observation = (
    f"It is a {game_parameters['players']} Player Hanabi game. The current player is {state.cur_player()}. "
    f"There is only {state.life_tokens()} life token; when it is 0, it's game over. There are "
    f"{state.information_tokens()} tokens to give a piece of information to other players. "
    f"{process_fireworks(state.fireworks())} The deck consists of {state.deck_size()}. "
    f"We can see other player cards except ours {final_player_infor_string}. The knowledge about our current cards is "
    f"{process_cards(knowledge_di[state.cur_player()])}."
  )

  return llm_observation


def run_game(game_parameters):
  """Play a game, selecting random actions."""

  def print_state(state):
    """Print some basic information about the state."""
    print("")
    print("Current player: {}".format(state.cur_player()))
    print(state)

    # Example of more queries to provide more about this state. For
    # example, bots could use these methods to to get information
    # about the state in order to act accordingly.
    print("### Information about the state retrieved separately ###")
    print("### Information tokens: {}".format(state.information_tokens()))
    print("### Life tokens: {}".format(state.life_tokens()))
    print("### Fireworks: {}".format(state.fireworks()))
    print("### Deck size: {}".format(state.deck_size()))
    print("### Discard pile: {}".format(str(state.discard_pile())))
    print("### Player hands: {}".format(str(state.player_hands())))
    print("")

  def print_observation(observation):
    """Print some basic information about an agent observation."""
    print("--- Observation ---")
    print(observation)

    print("### Information about the observation retrieved separately ###")
    print("### Current player, relative to self: {}".format(
        observation.cur_player_offset()))
    print("### Observed hands: {}".format(observation.observed_hands()))
    print("### Card knowledge: {}".format(observation.card_knowledge()))
    print("### Discard pile: {}".format(observation.discard_pile()))
    print("### Fireworks: {}".format(observation.fireworks()))
    print("### Deck size: {}".format(observation.deck_size()))
    move_string = "### Last moves:"
    for move_tuple in observation.last_moves():
      move_string += " {}".format(move_tuple)
    print(move_string)
    print("### Information tokens: {}".format(observation.information_tokens()))
    print("### Life tokens: {}".format(observation.life_tokens()))
    print("### Legal moves: {}".format(observation.legal_moves()))
    print("--- EndObservation ---")

  def print_encoded_observations(encoder, state, num_players):
    print("--- EncodedObservations ---")
    print("Observation encoding shape: {}".format(encoder.shape()))
    print("Current actual player: {}".format(state.cur_player()))
    for i in range(num_players):
      print("Encoded observation for player {}: {}".format(
          i, encoder.encode(state.observation(i))))
    print("--- EndEncodedObservations ---")
  game = pyhanabi.HanabiGame(game_parameters)
  print(game.parameter_string(), end="")
  obs_encoder = pyhanabi.ObservationEncoder(
      game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)

  state = game.new_initial_state()
  observation_di={}
  while not state.is_terminal():
    if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      state.deal_random_card()
      continue

    llm_observation = get_llm_observation(state, game_parameters)
    
    print(llm_observation)

    # observation = state.observation(state.cur_player())
    # # print_observation(observation)
    # try:
    #   print_encoded_observations(obs_encoder, state, game.num_players())
    # except RuntimeError as e:
    #   print(f"Error: {e}")
    legal_moves = state.legal_moves()
    print("")
    print("Number of legal moves: {}".format(len(legal_moves)))

    move = np.random.choice(legal_moves)
    if state.cur_player() not in observation_di:
      observation_di[state.cur_player()] = []
    observation_di[state.cur_player()].append([llm_observation,move ])
    print("Chose random legal move: {}".format(move))

    state.apply_move(move)

  print("")
  print("Game done. Terminal state:")
  print("")
  print(state)
  print("")
  print("score: {}".format(state.score()))


if __name__ == "__main__":
  # Check that the cdef and library were loaded from the standard paths.
  assert pyhanabi.cdef_loaded(), "cdef failed to load"
  assert pyhanabi.lib_loaded(), "lib failed to load"
  run_game({"players": 3, "random_start_player": True})
