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
  card_colors = {'Y': 'Yellow', 'B': 'Blue', 'R': 'Red', 'W': 'White', 'G': 'Green' , 'X': 'Unknown'}

  output = []
  for card in cards:
    card = str(card)
    color = card_colors[card[0]]
    number = card[1:]
    output.append(f"{color} card with number {number}")

  return ', '.join(output)


def process_fireworks(fireworks):
  print(fireworks)
  processed_firework_text = "The fireworks display includes "+str(fireworks[0]) +" Red firework, "+str(fireworks[1])+" Yellow fireworks, "+str(fireworks[2])+" Green firework, "+str(fireworks[3])+" White fireworks, and "+str(fireworks[4])+" Blue firework."
  return processed_firework_text


def knowledge(state_knowledge):
  k_di = {}
  state_knowledge = str(state_knowledge)

  pattern = re.compile(r'Hands:(.*?)Deck size:', re.DOTALL)

  # Find all matches in the input text
  matches = pattern.findall(state_knowledge)

  # Display the matches
  for match in matches:
    knowledge_hand = match.strip()

  lines = knowledge_hand.strip().split('\n')
  counter = 0
  for l in lines:
    # print('l',l)
    if l =="Cur player":
      continue
    if l == '-----':
      counter += 1
      continue
    if counter not in di:
      #   all_metrics[category] = {}
      print(counter)
      k_di[counter] = []
    k_di[counter].append(l.split(' || ')[1].split('|')[0])

  return k_di
def get_llm_observation(state):
  """
  SAMPLE FORMAT:
  It is a 2 Player Hanabi game. The current player is 1. There is only 1 life token when it is 0 its game over. There are 6 tokens to give a piece of information to other players. The fireworks display includes 0 Red firework, 0 Yellow fireworks, 2 Green firework, 0 White fireworks, and 0 Blue firework. The deck consists of 33. The other players hands are White card with number 1, Yellow card with number 1, Yellow card with number 4, Green card with number 4, Blue card with number 1. The knowledge about our current cards are Yellow card with number X, Unknown card with number X, Unknown card with number 2, Unknown card with number X, Unknown card with number X

  """
  other_player_info = state.player_hands()
  other_player_info_string = ""
  for i in range(0, len(other_player_info)):
    if i == state.cur_player():
      continue
    else:
      other_player_info_string += process_cards(other_player_info[i])

  knowledge_di = knowledge(state)

  llm_observation = "It is a " + str(game_parameters["players"]) + " Player Hanabi game. The current player is " + str(
    state.cur_player()) + ". There is only " + str(
    state.life_tokens()) + " life token when it is 0 its game over. There are " + str(
    state.information_tokens()) + " tokens to give a piece of information to other players. " + \
                    process_fireworks(state.fireworks()) + " The deck consists of " + str(
    state.deck_size()) + ". The other players hands are " + other_player_info_string + "." + \
                    " The knowledge about our current cards are " + process_cards(knowledge_di[state.cur_player()])
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

    llm_observation = get_llm_observation(state)

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
  run_game({"players": 2, "random_start_player": True})
