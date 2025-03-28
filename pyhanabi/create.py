import set_path
import torch
import rela
import hanalearn


def create_envs(
    num_env,
    seed,
    num_player,
    bomb,
    max_len,
    *,
    hand_size=5,
    random_start_player=1,
    num_color=5,
    num_rank=5,
    num_hint=8,
):
    games = []
    for game_idx in range(num_env):
        params = {
            "players": str(num_player),
            "seed": str(seed + game_idx),
            "bomb": str(bomb),
            "hand_size": str(hand_size),
            "random_start_player": str(random_start_player),
            "colors": str(num_color),
            "ranks": str(num_rank),
            "max_information_tokens": str(num_hint),
        }
        game = hanalearn.HanabiEnv(
            params,
            max_len,
            False,
        )
        games.append(game)
    return games


def flatten(s):
    if s == []:
        return s
    if isinstance(s[0], list):
        return flatten(s[0]) + flatten(s[1:])
    return s[:1] + flatten(s[1:])


def create_threads(num_thread, num_game_per_thread, actors, games):
    context = rela.Context()
    threads = []
    for thread_idx in range(num_thread):
        envs = games[
            thread_idx * num_game_per_thread : (thread_idx + 1) * num_game_per_thread
        ]
        # print ("envs size is: ", envs.size(), len(actors[thread_idx]))
        thread = hanalearn.HanabiThreadLoop(envs, actors[thread_idx], False)
        threads.append(thread)
        context.push_thread_loop(thread)
    print(
        "Finished creating %d threads with %d games and %d actors"
        % (len(threads), len(games), len(flatten(actors)))
    )
    return context, threads


class BatchRunner:
    def __init__(self, agent, device, methods):
        self.agent = agent.clone(device)
        self.runner = rela.BatchRunner(self.agent, device)
        # self.runner.set_log_freq(100)
        for method, bsz in methods.items():
            self.runner.add_method(method, bsz)

    def update_model(self, agent):
        self.runner.acquire_model_lock()
        self.agent.load_state_dict(agent.state_dict())
        self.runner.release_model_lock()

    def start(self):
        self.runner.start()

    def stop(self):
        self.runner.stop()

    def get(self):
        return self.runner


def create_model_runners(agent, devices, methods):
    runners = []
    for dev in devices:
        runner = BatchRunner(agent, dev, methods)
        runners.append(runner)
    return runners


def create_actor(actor_args, player_idx, model_runner, replay_buffer):
    actor_args["player_idx"] = player_idx
    actor_args["runner"] = model_runner
    actor_args["replay_buffer"] = replay_buffer
    return hanalearn.R2D2Actor(**actor_args)


class ActGroupBase:
    def __init__(self, devices, agent, method_batchsize):
        self.devices = devices.split(",")
        self.model_runners = create_model_runners(agent, self.devices, method_batchsize)

    @property
    def num_runners(self):
        return len(self.model_runners)

    def start(self):
        for runner in self.model_runners:
            runner.start()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner.update_model(agent)

class SelfplayActGroup(ActGroupBase):
    def __init__(
        self,
        devices,
        agent,
        seed,
        num_thread,
        num_game_per_thread,
        num_player,
        replay_buffer,
        actor_args,
        method_batchsize,
        *,
        explore_eps=None,
        boltzmann_t=None,
        llm_prior=None,
        pikl_lambda=0,
        pikl_beta=1,
    ):
        super().__init__(devices, agent, method_batchsize)

        self.actors = []
        self.flat_actors = []
        for i in range(num_thread):
            thread_actors = []
            for _ in range(num_game_per_thread):
                game_actors = []
                for k in range(num_player):
                    player_runner = self.model_runners[i % self.num_runners].get()
                    actor_args["seed"] = seed
                    seed += 1
                    actor = create_actor(actor_args, k, player_runner, replay_buffer)

                    if llm_prior is not None:
                        actor.set_llm_prior(llm_prior, pikl_lambda, pikl_beta)

                    if boltzmann_t is not None:
                        actor.set_boltzmann_t(boltzmann_t)

                    if explore_eps is not None:
                        actor.set_explore_eps(explore_eps)

                    game_actors.append(actor)
                    self.flat_actors.append(actor)

                # for k in range(num_player):
                #     partners = game_actors[:]
                #     partners[k] = None
                #     game_actors[k].set_partners(partners)

                thread_actors.append(game_actors)
            self.actors.append(thread_actors)

        print("Selfplay Group Created")



class ActGroup:
    def __init__(
        self,
        devices,
        agent,
        seed,
        num_thread,
        num_game_per_thread,
        num_player,
        explore_eps,
        boltzmann_t,
        method,
        sad,
        shuffle_color,
        hide_action,
        trinary,
        replay_buffer,
        multi_step,
        max_len,
        gamma,
        off_belief,
        belief_model,
    ):
        self.devices = devices.split(",")

        self.model_runners = []
        methods = {"act": 5000, "compute_priority": 100}
        if off_belief:
            methods["compute_target"] = 5000
        make_model_runners(agent, self.devices, self.model_runners, methods)
        self.num_runners = len(self.model_runners)

        self.off_belief = off_belief
        self.belief_model = belief_model
        self.belief_runner = None
        if belief_model is not None:
            self.belief_runner = []
            for bm in belief_model:
                print("add belief model to: ", bm.device)
                self.belief_runner.append(
                    rela.BatchRunner(bm, bm.device, 5000, ["sample"])
                )

        self.actors = []
        if method == "vdn":
            for i in range(num_thread):
                thread_actors = []
                for j in range(num_game_per_thread):
                    actor = hanalearn.R2D2Actor(
                        self.model_runners[i % self.num_runners],
                        seed,
                        num_player,
                        0,
                        explore_eps,
                        boltzmann_t,
                        True,
                        sad,
                        shuffle_color,
                        hide_action,
                        trinary,
                        replay_buffer,
                        multi_step,
                        max_len,
                        gamma,
                    )
                    seed += 1
                    thread_actors.append([actor])
                self.actors.append(thread_actors)
        elif method == "iql":
            for i in range(num_thread):
                thread_actors = []
                for j in range(num_game_per_thread):
                    game_actors = []
                    for k in range(num_player):
                        actor = hanalearn.R2D2Actor(
                            self.model_runners[i % self.num_runners],
                            seed,
                            num_player,
                            k,
                            explore_eps,
                            boltzmann_t,
                            False,
                            sad,
                            shuffle_color,
                            hide_action,
                            trinary,
                            replay_buffer,
                            multi_step,
                            max_len,
                            gamma,
                        )
                        if self.off_belief:
                            if self.belief_runner is None:
                                actor.set_belief_runner(None)
                            else:
                                actor.set_belief_runner(
                                    self.belief_runner[i % len(self.belief_runner)]
                                )
                        seed += 1
                        game_actors.append(actor)
                    for k in range(num_player):
                        partners = game_actors[:]
                        partners[k] = None
                        game_actors[k].set_partners(partners)
                    thread_actors.append(game_actors)
                self.actors.append(thread_actors)
        print("ActGroup created")

    def start(self):
        for runner in self.model_runners:
            runner.start()

        if self.belief_runner is not None:
            for runner in self.belief_runner:
                runner.start()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner.update_model(agent)

def make_model_runners(agents, devices, runners, methods):
    if not isinstance(agents, list):
        agents = [agents]
    for dev in devices:
        for agent in agents:
            runner = rela.BatchRunner(agent.clone(dev), dev)
            for method, sample_limit in methods.items():
                runner.add_method(method, sample_limit)
            runners.append(runner)


