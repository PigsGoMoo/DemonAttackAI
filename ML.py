import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

env = gym.make('DemonAttack-v0')
env.reset()


# Random game to see what the dataset will look like
def play_game(amt: int):
    previous_lives = {"ale.lives": 4}
    for step_index in range(amt):
        env.render()
        action = env.action_space.sample()
        # print(env.step(action))
        observation, reward, done, info = env.step(action)
        # print("Step {}:".format(step_index))
        # print("action: {}".format(action))
        # print("observation: {}".format(observation))
        print("reward: {}".format(reward))
        # print("done: {}".format(done))
        print("info: {}".format(info['ale.lives']))

        if info['ale.lives'] < previous_lives['ale.lives']:
            print("Life lost :(")

        print("=" * 40)
        previous_lives = info
        if done:
            break


# play_game(1000)

goal = 3000
score_requirement = 150
initial_games = 10
# svd = PCA(n_components=3)


def encode_data(data):
    output = []
    if data == 1:
        output = [0, 1, 0, 0, 0, 0]
    elif data == 0:
        output = [1, 0, 0, 0, 0, 0]
    elif data == 2:
        output = [0, 0, 1, 0, 0, 0]
    elif data == 3:
        output = [0, 0, 0, 1, 0, 0]
    elif data == 4:
        output = [0, 0, 0, 0, 1, 0]
    elif data == 5:
        output = [0, 0, 0, 0, 0, 1]

    return output


# Gather our data
def prep_model_data():
    # Initialize our training data and scores array here
    training_data = []
    accepted_scores = []

    # Play the game initial_games times
    for game_index in range(initial_games):
        # Initialize our variables for score, memory, and previous step
        score = 0
        game_memory = []
        previous_observation = []
        previous_lives = {"ale.lives": 4}
        print("Game number: {}/{}".format(game_index, initial_games))
        # Start playing the game playing until you hit the goal or lose
        for step_index in range(goal):
            # According to the docs, there are 6 available actions for
            # this game, numbered 0-5. So we will pick a random action
            # and execute it here
            env.render()
            action = random.randrange(0, 6)

            # Grab our state info from the game
            observation, reward, done, info = env.step(action)

            # Check to see if we've taken an action before
            if len(previous_observation) > 0:
                # Add our previous action and the result to memory
                # game_memory.append([previous_observation, info['ale.lives'], action])
                game_memory.append([previous_observation, action])

                if info['ale.lives'] < previous_lives['ale.lives']:
                    print("Life lost!")
                    score -= 20

            # Update the score and observation
            previous_lives = info
            previous_observation = observation
            score += reward

            # Exit loop if we lost
            if done:
                print("Game over!")
                break

        # See if the steps we took in this run-through was able to beat
        # the score we want it to beat for recording
        if score >= score_requirement:
            # Add it to accepted scores if so
            accepted_scores.append(score)
            # The data in game memory needs to be One Hot Encoded now
            for data in game_memory:
                output = encode_data(data[1])
            # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
            #                        remainder='drop')
            #
            # encoded = ct.fit_transform(game_memory)

            # round_data = np.array(ct.fit_transform(game_memory))

            # print(encoded)

            # formatted_round_data = []
            # for round in round_data:
            #     formatted_round_data.append([round[2], [round[0], round[1]]])
            #
            # for round2 in formatted_round_data:
            #     training_data.append(round2)

            # One Hot Encoding doesn't accept multiple nested lists, so
            # I have to manually encode it...

            # Add the One Hot Encoded data to training_data
            #     print("*" * 70)
            #     print(np.reshape(data[0], -1))
            #     print("*" * 70)
                data[0] = np.reshape(data[0], -1)
                # print(np.append(data[0], data[1]))
                # training_data.append([np.append(data[0], data[1]), output])
                training_data.append([data[0], output])

        print("Score achieved: {}".format(score))
        # Reset the game environment and play again
        env.reset()

    # Print out accepted scores and see game data we'll be feeding to bot
    print("{} scores accepted out of {} games played".format(len(accepted_scores), initial_games))
    print(accepted_scores)
    # print("Game memory: {}".format(game_memory))
    return training_data


print("Preparing model data based on {} games".format(initial_games))
training_data = prep_model_data()
print("We are working with {} training data".format(len(training_data)))
# print(training_data)


# Build our ANN model

def build_model(input_size: int, output_size: int):
    ann = Sequential()
    ann.add(Dense(1024, input_dim=input_size, activation='relu'))
    ann.add(Dense(256, input_dim=input_size, activation='relu'))
    ann.add(Dense(64, activation='relu'))
    ann.add(Dense(output_size, activation='linear'))
    ann.compile(loss='mse', optimizer=Adam(), metrics=['categorical_crossentropy'])
    ann.summary()
    return ann


# Train our model

def train_model(training_data: list):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    # print(training_data[0][0])
    # print(len(x))
    # print("=" * 50)
    # print(len(y))
    # x = np.asarray(x).astype(np.int)
    model = build_model(input_size=len(x[0]), output_size=len(y[0]))
    model.fit(x, y, epochs=20)
    return model


print("Training our AI")
trained_model = train_model(training_data)


def run_taught_AI():
    scores = []
    choices = []
    for each_game in range(1):
        score = 0
        prev_obs = []
        print("==" * 50)
        print("=" + " " * 45 + "GAME #{}".format(each_game) + " " * 45 + "=")
        print("==" * 50)
        prev_info = {'ale.lives': 4}
        for step_index in range(10000):
            env.render()
            if len(prev_obs) == 0:
                action = random.randrange(0,6)
                print("Taking a random action...")
            else:
                print("Calculating next move...")
                x = np.reshape(prev_obs, -1)
                # x = np.append(x, prev_info['ale.lives'])
                # print('Reshaped: {}'.format(x))
                next_move = trained_model.predict(x.reshape(-1, len(x)))
                print("Next move: {}".format(next_move))
                action = np.argmax(next_move)
                # action = np.argmax(trained_model.predict(np.reshape(prev_obs, -1)))
                print("Taking a calculated action: {}\n"
                      "This is step number: {}".format(action, step_index))

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_info = info
            prev_obs = new_observation
            score += reward
            if done:
                break

        env.reset()
        scores.append(score)

    print(scores)
    print('Average Score:', sum(scores)/len(scores))
    print('choice 0:{}  choice 1:{}  choice 2:{}  choice 3:{}  choice 4:{}  choice 5:{}'
          .format(choices.count(0)/len(choices), choices.count(1)/len(choices),
                  choices.count(2)/len(choices), choices.count(3)/len(choices),
                  choices.count(4)/len(choices), choices.count(5)/len(choices)))


run_taught_AI()
