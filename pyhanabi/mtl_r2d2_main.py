import time
import os
import sys
import argparse
import pprint
import pickle
import json
import numpy as np
import torch

from create import (
    create_envs,
    create_threads,
    SelfplayActGroup,
)
from eval import evaluate
import common_utils
import rela
import hanalearn  # type: ignore
import r2d2
import utils

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--config", type=str, default=None)

    # training setup related
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--save_per", type=int, default=50)
    parser.add_argument("--load_model", type=str, default="None")
    parser.add_argument("--seed", type=str, default="a")
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--act_device", type=str, default="cuda:0")
    parser.add_argument("--actor_sync_freq", type=int, default=10)
    parser.add_argument("--target_data_ratio", type=float, default=None, help="train/gen")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)

    # algo setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--num_eps", type=float, default=80)
    parser.add_argument("--aux_weight", type=float, default=0)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument("--multi_step", type=int, default=1)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)
    parser.add_argument("--vdn", type=int, default=0)
    parser.add_argument("--shuffle_color", type=int, default=0)

    # optim setting
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--batchsize", type=int, default=128)

    # model setting
    parser.add_argument("--net", type=str, default="publ-lstm", help="publ-lstm/lstm/ffwd")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    # replay/data settings
    parser.add_argument("--replay_buffer_size", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")
    parser.add_argument("--burn_in_frames", type=int, default=1000)
    parser.add_argument("--eval_freq", type=int, default=5)

    # llm setting
    parser.add_argument("--llm_prior", type=str, default=None)
    parser.add_argument("--pikl_lambda", type=float, default=0.0)
    parser.add_argument("--pikl_anneal_per", type=int, default=10)
    parser.add_argument("--pikl_anneal_rate", type=float, default=0.5)
    parser.add_argument("--pikl_beta", type=float, default=1)
    parser.add_argument("--llm_noise", type=float, default=0.0)
    parser.add_argument("--llm_noise_seed", type=int, default=97)

    # game config
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--hand_size", type=int, default=5)
    parser.add_argument("--num_color", type=int, default=5)
    parser.add_argument("--num_rank", type=int, default=5)
    parser.add_argument("--num_hint", type=int, default=8)
    parser.add_argument("--bomb", type=int, default=0)

    # debug
    parser.add_argument("--do_eval", type=int, default=1)
    parser.add_argument("--wandb", type=str, default=1)

    parser.add_argument("--update_freq_text_enc", type=str, default=1)
    parser.add_argument("--lm_weights", type=str, default=1)
    parser.add_argument("--num_of_additional_layer", type=str , default=1)
    parser.add_argument("--num_lm_layer", type=str , default=1)
    parser.add_argument("--lora_dim", type=str , default=0)

    # slurm job arguments to relaunch the program
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=20)

    args = parser.parse_args()
    args = common_utils.maybe_load_config(args)
    args.update_freq_text_enc = int(args.update_freq_text_enc)
    args.wandb = int(args.wandb)
    args.num_lm_layer = int(args.num_lm_layer)
    args.lora_dim = int(args.lora_dim)

    args.seed = utils.get_seed(args.seed)
    #
    # if args.load_model:
    #     save_path= args.load_model.split('/')[-2] + '/' + args.load_model.split('/')[-1]
    #     args.save_dir = (args.save_dir + f'loaded_{save_path}_np_{args.num_player}'
    #                      + f'_text_enc_{args.lm_weights}_s_{args.seed}')
    # else:

    # args.save_dir = (args.save_dir + f'_np_{args.num_player}' + f'_text_enc_{args.lm_weights}_s_{args.seed}')
    return args


def train(args):
    common_utils.set_all_seeds(args.seed)
    if args.wandb:
        wandb.init(project='r2d2_drrn', entity='sarath-chandar', config=args)

    logger_path = os.path.join(args.save_dir, f"train_"+str(args.start_epoch)+'.log')
    sys.stdout = common_utils.Logger(logger_path, print_to_stdout=True)
    pprint.pprint(vars(args))
    utils.print_gpu_info()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_device = args.train_device
    act_device = args.act_device
    games = []
    if args.num_player==6:
        players_list=[2,3,4,5]
    else:
        players_list = [args.num_player]
    # players_list= [2]
    for num_player in players_list:
        games.append(create_envs(
            args.num_thread * args.num_game_per_thread,
            args.seed,
            num_player,
            args.bomb,
            args.max_len,
            hand_size=args.hand_size,
            num_color=args.num_color,
            num_rank=args.num_rank,
            num_hint=args.num_hint,
        ))

    agent = r2d2.R2D2Agent(
        args.vdn,
        args.multi_step,
        args.gamma,
        train_device,
        1, # in_dim
        args.rnn_hid_dim,
        games[0][0].num_action(),
        args.net,
        args.num_lstm_layer,
        args.lm_weights,
        args.num_player, # num_players
        args.num_of_additional_layer,
        args.num_lm_layer,
        args.lora_dim,
        off_belief=False,

    )
    print(agent)


    if args.load_model and args.load_model != "None":
        print("*****loading pretrained model*****")
        print(args.load_model)
        print(args.load_model.replace('epoch','epoch_optim').replace('.pthw','.pth'))
        online_net = utils.get_agent_online_network(agent, False)
        utils.load_weight(online_net, args.load_model, train_device)
        optim = torch.optim.Adam(online_net.parameters(), lr=args.lr, eps=args.eps)
        optim.load_state_dict(torch.load(args.load_model.replace('epoch','epoch_optim').replace('.pthw','.pth') ,map_location=train_device))

        head, tail = args.load_model.rsplit('/', 1)  # head is the directory, tail is the filename
        # # Replace the epoch in the filename and change the extension
        # filename_buffer = f'replay_buffer.pth'
        # # Construct the new path
        # replay_buffer_path = f'{head}/{filename_buffer}'
        # print('replay_buffer_path', replay_buffer_path)
        #
        # replay_buffer = torch.load(replay_buffer_path)

        print("***************done***************")
    else:
        online_net = utils.get_agent_online_network(agent, False)
        optim = torch.optim.Adam(online_net.parameters(), lr=args.lr, eps=args.eps)

    replay_buffer = rela.RNNReplay(  # type: ignore
        args.replay_buffer_size,
        args.seed,
        args.prefetch,
    )

    saver = common_utils.TopkSaver(args.save_dir, 5)


    explore_eps = utils.generate_explore_eps(args.act_base_eps, args.act_eps_alpha, args.num_eps)
    eps_str = [[f"\n{eps:.9f}", f"{eps:.9f}"][i % 5 != 0]  for i, eps in enumerate(explore_eps)]
    print("explore eps:", ", ".join(eps_str))
    print("avg explore eps:", np.mean(explore_eps))
    act_groups = []
    contexts = []
    for i,p in enumerate(players_list):
        # making input arguments
        act_group_args = {
            "devices": act_device,
            "agent": agent,
            "seed": args.seed,
            "num_thread": args.num_thread,
            "num_game_per_thread": args.num_game_per_thread,
            "num_player": p,
            "replay_buffer": replay_buffer,
            "method_batchsize": {"act": 5000},
            "explore_eps": explore_eps,
        }

        act_group_args["actor_args"] = {
            "num_player": p,
            "vdn": args.vdn,
            "sad": False,
            "shuffle_color": args.shuffle_color,
            "hide_action": False,
            "trinary": hanalearn.AuxType.Trinary,
            "multi_step": args.multi_step,
            "seq_len": args.max_len,
            "gamma": args.gamma,
        }

        llm_prior = None

        act_group_type = SelfplayActGroup
        act_groups.append(act_group_type(**act_group_args))
        contexts.append(create_threads(
            args.num_thread,
            args.num_game_per_thread,
            act_groups[i].actors,
            games[i],
        )[0])

        act_groups[i].start()
        contexts[i].start()

    train_eval_metrics_path = args.save_dir + '/train_eval_metrics.json'
    if os.path.exists(train_eval_metrics_path):
        # Load the dictionary from the JSON file
        with open(train_eval_metrics_path, 'r') as file:
            train_eval_metrics = json.load(file)
        print(f"Dictionary loaded from {train_eval_metrics_path}")
    else:
        # Create a new dictionary
        train_eval_metrics = {}
        print(f"No file found at {train_eval_metrics_path}. Created a new dictionary.")

    if args.do_eval and args.start_epoch==1:
        wandb_metrics = {}
        for i,p in enumerate(players_list):
            score, perfect, *_ = evaluate(
                [agent],
                10,
                np.random.randint(100000),
                args.bomb,
                num_player=p,
                pikl_lambdas=None if llm_prior is None else [args.pikl_lambda],
                pikl_betas=None if llm_prior is None else [args.pikl_beta],
                llm_priors=None if llm_prior is None else [llm_prior],
                hand_size=args.hand_size,
                num_color=args.num_color,
                num_rank=args.num_rank,
                num_hint=args.num_hint,
            )
            perfect *= 100
            print(
                f"Eval(epoch 0): {p}-player score: {score}, perfect: {perfect}"
            )

            train_eval_metrics[0] = [perfect, score ,0 , 0, 0]

            if args.wandb:
                wandb_metrics.update({f"{p}-player_score": score, f"{p}-player_perfect": perfect})
        if args.wandb:
            wandb_metrics.update({"epoch": 0})
            wandb.log(wandb_metrics)
    else:
        score, perfect = None, None


    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)
    print("Success, Done")
    print("=" * 100)
    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()
    sleep_time = 0

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        # if (args.pikl_lambda > 0
        #     and epoch > 0
        #     and args.pikl_anneal_per > 0
        #     and epoch % args.pikl_anneal_per == 0
        # ):
        #     args.pikl_lambda *= args.pikl_anneal_rate
        #     for actor in act_group.flat_actors:
        #         actor.update_llm_lambda([args.pikl_lambda])
        print("start_epoch",args.start_epoch)
        print(f"EPOCH: {epoch}, pikl_lambda={args.pikl_lambda}")
        print(common_utils.get_mem_usage("(epoch start)"))
        tachometer.start()
        stat.reset()
        stopwatch.reset()

        for batch_idx in range(args.epoch_len):
            with stopwatch.time("sync & update"):
                num_update = batch_idx + epoch * args.epoch_len
                if num_update % args.num_update_between_sync == 0:
                    agent.sync_target_with_online()
                if num_update % args.actor_sync_freq == 0:
                    for i,p in enumerate(players_list):
                        act_groups[i].update_model(agent)
                torch.cuda.synchronize()

            with stopwatch.time("sample data"):
                batch = replay_buffer.sample(args.batchsize, train_device)

            with stopwatch.time("forward"):
                update_text_encoder = False
                if num_update % args.update_freq_text_enc == 0:
                    update_text_encoder = True
                loss = agent.loss(batch, args.aux_weight, stat)
                loss = loss.mean()

            with stopwatch.time("backward"):
                loss.backward()

            with stopwatch.time("synchronize"):
                torch.cuda.synchronize()

            with stopwatch.time("optim step"):
                g_norm = torch.nn.utils.clip_grad_norm_(online_net.parameters(), args.grad_clip)
                optim.step()
                optim.zero_grad()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            with stopwatch.time("sleep"):
                if sleep_time > 0:
                    time.sleep(sleep_time)

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)

        with stopwatch.time("eval & others"):
            count_factor = 1 if not args.vdn else args.num_player
            new_sleep_time, train_steps, trajectories, num_of_actions = tachometer.lap(
                replay_buffer,
                args.epoch_len * args.batchsize,
                count_factor,
                num_batch=args.epoch_len,
                target_ratio=args.target_data_ratio,
                current_sleep_time=sleep_time
            )
            sleep_time = 0.6 * sleep_time + 0.4 * new_sleep_time
            print(
                f"Sleep info: new_sleep_time: {int(1000 * new_sleep_time)} MS, "
                f"actual_sleep_time: {int(1000 * sleep_time)} MS"
            )

            stat.summary(epoch)

            if args.do_eval and (epoch+1) % args.eval_freq == 0:

                for i,p in enumerate(players_list):
                    contexts[i].pause()

                print(common_utils.get_mem_usage("(before eval)"))
                wandb_metrics = {}
                for i,p in enumerate(players_list):
                    score, perfect, *_ = evaluate(
                        [agent],
                        10,
                        np.random.randint(100000),
                        args.bomb,
                        num_player=p,
                        pikl_lambdas=None if llm_prior is None else [args.pikl_lambda],
                        pikl_betas=None if llm_prior is None else [args.pikl_beta],
                        llm_priors=None if llm_prior is None else [llm_prior],
                        hand_size=args.hand_size,
                        num_color=args.num_color,
                        num_rank=args.num_rank,
                        num_hint=args.num_hint,
                    )
                    perfect *= 100
                    print(
                        f"Eval(epoch {epoch}): {p}-player score: {score}, perfect: {perfect}"
                    )
                    if args.wandb:
                        wandb_metrics.update({f"{p}-player_score": score, f"{p}-player_perfect": perfect})
                if args.wandb:
                    wandb_metrics.update({"epoch": epoch, "episodes(num_buffer)": tachometer.num_buffer, "train_steps": tachometer.num_train,
                               "action_steps": tachometer.num_of_actions})
                    wandb.log(wandb_metrics)
                print(common_utils.get_mem_usage("(after eval)"))

                for i,p in enumerate(players_list):
                    contexts[i].resume()
                #
                # train_eval_metrics['perfect'] = perfect
                # train_eval_metrics['score'] = score
                # train_eval_metrics['train_steps'] = train_steps
                # train_eval_metrics['num_of_traj'] = trajectories
                # train_eval_metrics['num_of_actions'] = num_of_actions
                # train_eval_metrics['epoch'] = epoch


            else:
                score, perfect = None, None

            train_eval_metrics[epoch] = [perfect, score, train_steps, trajectories, num_of_actions]

            force_save = f"epoch{epoch + 1}" if (epoch + 1) % args.save_per == 0 else None
            force_save_optim = f"epoch_optim{epoch + 1}" if (epoch + 1) % args.save_per == 0 else None
            model_saved = saver.save(
                online_net.state_dict(),optim.state_dict(),replay_buffer, score, force_save_name=force_save , force_save_name_optim= force_save_optim,force_replay_buffer_path= 'replay_buffer', config=vars(args)
            )

            # torch.save(optim.state_dict(), '/home/mila/a/arjun.vaithilingam-sudhakar/scratch/final_hanabi_checkpoint/drrn_baselines/optimizer.pth')
            # print(
            #     f"Eval(epoch {epoch+1}): score: {score}, perfect: {perfect}, model saved: {model_saved}"
            # )


        stopwatch.summary()
        print("=" * 100)

    # check the file exist
    # previously, if exists use it or create a new dict
    # overwrite on the same file

    with open(train_eval_metrics_path, 'w') as file:
        json.dump(train_eval_metrics, file)
        print(f"Dictionary saved to {train_eval_metrics_path}")
    # force quit, "nicely"
    os._exit(0)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # type: ignore
    args = parse_args()

    train(args)
