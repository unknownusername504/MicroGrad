# Use the ring reduction to train a copy of the model on a mini-batch of data
# and collect the results.
from micrograd.models.cartpole.simple_cartpole import CartPole

from micrograd.scheduler.ring_reduce_real import RingCoordinator


def train_models_minibatched():
    num_rounds = 20
    num_episodes_per_round = 500
    # Number of cores to split across
    num_splits = 8
    # Num rounds before reduce
    num_rounds_reduce = 2
    base_cart_pole = CartPole(do_load_model=True, do_save_model=True)
    ring_coordinator = RingCoordinator(num_splits, base_cart_pole)
    for round_num in range(num_rounds // num_rounds_reduce):
        print(f"Training round {round_num}")
        # Sync and collect the results from the runner and run all reduce to update the models across all cores
        ring_coordinator.run_ring(num_rounds_reduce, num_episodes_per_round)
        base_cart_pole.test(20)
    ring_coordinator.stop_ring()


if __name__ == "__main__":
    print("Starting training and testing")
    train_models_minibatched()
    print("Training and testing finished")
