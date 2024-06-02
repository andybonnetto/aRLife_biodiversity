"""
    Contains the main loop, used to run and visualize the automaton dynamics.
"""

import os
from Camera import Camera
from CA1D import CA1D, GeneralCA1D
from CA2D import CA2D
from Baricelli import Baricelli1D, Baricelli2D
from utils import launch_video, add_frame, save_image
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
import torch
import numpy as np
import scipy.fftpack
import scipy.ndimage as ndi
from creature import Creature, Encyclopedia
import argparse
import matplotlib.pyplot as plt


def apply_decomposition(world_states, W, H, patch_size=10, mode="PCA", n_components=3):
    decomp_world_states = torch.tensor(np.array(world_states))  # 3 x 400,300,3
    decomp_world_states = decomp_world_states[:, :, :, 0]  # 3 x 400,300,3
    patches = []
    for i in range(0, W, patch_size):
        for j in range(0, H, patch_size):
            patch = decomp_world_states[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    world_state = torch.stack(patches)
    world_state = torch.reshape(
        world_state, (world_state.shape[0], world_state.shape[1], -1)
    )
    world_state = torch.reshape(world_state, (world_state.shape[0], -1))
    world_state = torch.nan_to_num(world_state, nan=0.0)

    labels = run_decompos(world_state, mode=mode, n_components=n_components)
    return labels


def run_decompos(arr, mode="PCA", n_components=3):
    model = {
        "PCA": PCA(n_components=n_components),
        "ICA": FastICA(n_components=n_components),
        "Kmeans": KMeans(n_clusters=n_components),
    }.get(mode)
    model.fit(arr)
    if mode == "Kmeans":
        labels = model.labels_
    else:
        features = model.transform(arr)
        labels = features.argmax(axis=1)
    return labels


def creature_search(
    world_states: list, W: int, H: int, patch_size: int = 10, halved: bool = False
):
    """Search for creatures in the world state using a sliding window approach and border conditions"""
    world_state = np.mean(world_states, axis=0)
    decomp_world_states = torch.tensor(np.array(world_state))  # 400,300,3
    decomp_world_states = decomp_world_states[:, :, 0]  # 400,300
    patches = []
    step = int(patch_size / 2) if halved else patch_size

    for i in range(0, W - patch_size, step):
        for j in range(0, H - patch_size, step):
            patch = decomp_world_states[i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    world_state = torch.stack(patches)
    border_indices = np.array(
        [
            (i, j)
            for i in range(patch_size)
            for j in range(patch_size)
            if i == 0 or i == patch_size - 1 or j == 0 or j == patch_size - 1
        ]
    )
    cond_1 = (
        np.sum(
            world_state[:, border_indices[:, 0], border_indices[:, 1]].numpy(), axis=1
        )
        == 0
    )
    cond_2 = np.sum(world_state.reshape(world_state.shape[0], -1).numpy(), axis=1) > 0
    bool_creatures = np.logical_and(cond_1, cond_2)
    loc_arr = np.zeros_like(decomp_world_states)

    indices_W = np.arange(0, W - patch_size, step)
    indices_H = np.array(
        [np.arange(0, H - patch_size, step)] * len(indices_W)
    ).flatten()
    indices_W = np.repeat(indices_W, len(np.arange(0, H - patch_size, step)))

    loc_arr[indices_W, indices_H] = bool_creatures
    loc_x = indices_W[bool_creatures]
    loc_y = indices_H[bool_creatures]
    return (world_state[bool_creatures], loc_arr, loc_x, loc_y)


def phash(image, hash_size=32, lp=8):
    # Resize
    image = np.resize(image, (hash_size, hash_size))
    # Compute DCT
    dct = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0), axis=1)
    # Retain lower frequencies
    dct_lowfreq = dct[:lp, :lp]
    # Median value
    median = np.median(dct_lowfreq)
    # Set bits based on median value
    hash_value = 0
    for i in range(lp):
        for j in range(lp):
            hash_value |= (dct_lowfreq[i, j] > median) << (i * lp + j)
    return hash_value


def plot_num_creatures(
    num_creatures: list,
    output_dir: str,
    filename: str,
    title: str = "Number of creatures per frame",
):
    """Plot the number of creatures per frame"""
    num_creatures = np.array(num_creatures)
    plt.figure(figsize=(15, 10))
    if len(num_creatures.shape) > 1:
        # Plot as image
        plt.imshow(
            num_creatures.T, aspect="auto", interpolation="nearest", cmap="inferno_r"
        )
        plt.colorbar()
    else:
        plt.plot(num_creatures)
    plt.xlabel("Frame")
    plt.ylabel("Number of creatures")
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def recenter_creatures(creatures: list, patch_size: int, zoom_ratio: int = 1):
    """Recenter the creatures in the patch"""
    centered_creatures = np.zeros(
        (len(creatures), patch_size * zoom_ratio, patch_size * zoom_ratio)
    )
    for i, creat in enumerate(np.array(creatures)):
        pca = PCA(n_components=2)
        cy, cx = ndi.center_of_mass(creat)
        dim_x, dim_y = np.where(creat > 0)
        if len(dim_x) > 1:
            list_2d = np.concatenate(
                (dim_x.reshape(-1, 1), dim_y.reshape(-1, 1)), axis=1
            )
            list_2d = list_2d - np.repeat(
                np.array([cx, cy])[np.newaxis, :], len(list_2d), axis=0
            )
            pca.fit(list_2d)
            # realigned_2d = np.floor(np.dot(pca.components_, list_2d.T) + np.repeat(np.array([cx, cy])[:,np.newaxis], len(list_2d), axis = 1)).astype(np.int32).T
            realigned_2d = (
                np.floor(
                    np.dot(pca.components_, list_2d.T) + patch_size * zoom_ratio / 2
                )
                .astype(np.int32)
                .T
            )
            # indices = np.where((realigned_2d >= 0) | (realigned_2d < patch_size*zoom_ratio -1))[0]
            # realigned_2d = realigned_2d[indices]
            # realigned_2d = realigned_2d[np.where(realigned_2d < patch_size)[0]]
            centered_creatures[i][realigned_2d[:, 0], realigned_2d[:, 1]] = creat[
                dim_x[:], dim_y[:]
            ]
        # else: #TODO adapt zoom ratio
        # centered_creatures[i] = np.roll(creat, (int(patch_size/2 - cy),int(patch_size/2 -  cx)), axis = (0,1))
    return centered_creatures


def plot_encyclopedia(pokedex: Encyclopedia, output_dir: str, filename: str):
    """Plot the creatures in the encyclopedia"""
    _, axs = plt.subplots(
        int(np.sqrt(len(pokedex.creatures))) + 1,
        int(np.sqrt(len(pokedex.creatures))) + 1,
        figsize=(15, 15),
    )
    axs = axs.flatten()
    for i, creat in enumerate(pokedex.creatures):
        axs[i].imshow(creat.creature_data, cmap="cubehelix")
        axs[i].axis("off")
    for k in range(i, len(axs)):
        axs[k].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def run_simulation(
    W,
    H,
    fps,
    recording,
    launch_vid,
    patch_size,
    max_frames,
    random,
    classify,
    dec_mode,
    b_rule,
    s_rule,
    headless,
):

    output_dir = os.path.join(
        args.output_dir,
        f"rule_b{b_rule}_s{s_rule}_patch{patch_size}_duration{max_frames}",
    )
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not headless:
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((W, H), flags=pygame.SCALED | pygame.RESIZABLE)
        clock = pygame.time.Clock()

    running = True
    stopped = not headless
    camera = Camera(W, H)

    auto = CA2D((H, W), b_num=b_rule, s_num=s_rule, random=random)

    # Additional variables
    ws_buff = []
    zoom_ratio = 3
    pokedex = Encyclopedia(patch_size=patch_size, zoom_ratio=zoom_ratio)
    frame_count = 0
    detection_num = []
    creature_num = []
    pokedex_size = []
    creature_diversity_cumulative = []
    creature_diversity_variation = []
    creature_diversity = []
    writer = None

    while running:
        if frame_count >= max_frames:
            running = False

        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                camera.handle_event(event)  # Handle the camera events

                if event.type == pygame.KEYDOWN:
                    if (
                        event.key == pygame.K_SPACE
                    ):  # Press 'SPACE' to start/stop the automaton
                        stopped = not (stopped)
                    if (event.key == pygame.K_q) or (frame_count >= max_frames):
                        running = False
                    if event.key == pygame.K_r:  # Press 'R' to start/stop recording
                        recording = not recording
                        if not launch_vid and writer is not None:
                            launch_vid = True
                            writer.release()
                    if event.key == pygame.K_p:
                        save_image(auto.worldmap)

                auto.process_event(event, camera)  # Process the event in the automaton

        if not stopped:
            auto.step()  # step the automaton
            frame_count += 1

        auto.draw()  # draw the worldstate

        world_state = auto.worldmap
        ws_buff.append(world_state)
        if not frame_count % 3:

            creatures, loc_arr, loc_x, loc_y = creature_search(
                ws_buff, W, H, patch_size=patch_size, halved=True
            )
            detection_num.append(len(creatures))
            pokedex_size.append(pokedex.get_num_types())

            if len(creatures) > 2:
                # Recenter creatures
                centered_creatures = recenter_creatures(
                    creatures, patch_size, zoom_ratio=zoom_ratio
                )
                temp = pokedex.get_num_creatures()
                temp_div = pokedex.get_num_creatures_per_type()
                for i, creature in enumerate(centered_creatures):
                    c = Creature(creature, creature_pos=[loc_x[i], loc_y[i]])
                    pokedex.update(c)
                creature_num.append(pokedex.get_num_creatures() - temp)
                creature_diversity_variation.append(
                    pokedex.get_num_creatures_per_type()
                    - np.pad(
                        temp_div,
                        (0, len(pokedex.get_num_creatures_per_type()) - len(temp_div)),
                    )
                )
                creature_diversity_cumulative.append(
                    pokedex.get_num_creatures_per_type()
                )
                creature_diversity.append([len(track) for track in pokedex.tracks])
            pokedex.update_tracks()
            ws_buff = []  # Empty the buffer

        if not headless:
            surface = pygame.surfarray.make_surface(world_state)

        if recording:
            if launch_vid:  # If the video is not launched, we create it
                launch_vid = False
                writer = launch_video((H, W), fps, "mp4v")
            add_frame(
                writer, world_state
            )  # (in the future, we may add the zoomed frame instead of the full frame)

        if not headless:
            # Clear the screen
            screen.fill((0, 0, 0))

            # Draw rectangle for creatures
            if args.plot_detection:
                if len(np.where(loc_arr)[0]):
                    pos_x = np.where(loc_arr)[0]
                    pos_y = np.where(loc_arr)[1]
                    for x, y in zip(pos_x, pos_y):
                        rect = pygame.Rect((x, y), (patch_size, patch_size))
                        pygame.draw.rect(surface, (255, 0, 0), rect, 1)

            if args.plot_creatures:
                for track_creat in pokedex.tracks:
                    for track in track_creat:
                        rect = pygame.Rect(
                            (track[0], track[1]), (patch_size, patch_size)
                        )
                        pygame.draw.rect(surface, (255, 0, 0), rect, 1)

            # Draw the scaled surface on the window
            zoomed_surface = camera.apply(surface)

            screen.blit(zoomed_surface, (0, 0))

            # blit a red circle down to the left when recording
            if recording:
                pygame.draw.circle(screen, (255, 0, 0), (15, H - 15), 5)

            # Update the screen
            pygame.display.flip()

            clock.tick(90)  # limits FPS to 60

    if not headless:
        pygame.quit()

    if classify:
        all_creat = np.array([c.creature_data for c in pokedex.creatures])
        labels = run_decompos(
            all_creat.reshape(all_creat.shape[0], -1),
            mode=dec_mode,
            n_components=min(len(creatures), 8),
        )
        for creat, label in zip(pokedex.creatures.values(), labels):
            creat.creature_type = label
        pokedex.save_creatures(output_dir=output_dir)
    creature_diversity_variation = np.array(
        [
            np.pad(div, (0, len(creature_diversity_variation[-1]) - len(div)))
            for div in creature_diversity_variation
        ]
    )
    creature_diversity_cumulative = np.array(
        [
            np.pad(div, (0, len(creature_diversity_cumulative[-1]) - len(div)))
            for div in creature_diversity_cumulative
        ]
    )
    creature_diversity = np.array(
        [
            np.pad(div, (0, len(creature_diversity[-1]) - len(div)))
            for div in creature_diversity
        ]
    )

    pokedex.save(output_dir=output_dir)

    make_all_plots(
        creature_diversity,
        creature_diversity_variation,
        creature_diversity_cumulative,
        detection_num,
        creature_num,
        pokedex_size,
        pokedex,
        frame_count,
        output_dir,
    )


def make_all_plots(
    creature_diversity: np.array,
    creature_diversity_variation: np.array,
    creature_diversity_cumulative: np.array,
    detection_num: list,
    creature_num: list,
    pokedex_size: list,
    pokedex: Encyclopedia,
    frame_count: int,
    output_dir: str,
) -> None:
    """Make all the plots for the simulation results"""

    plot_num_creatures(
        creature_diversity,
        output_dir=output_dir,
        filename="creature_diversity.png",
        title="Creature diversity per frame",
    )

    plot_num_creatures(
        creature_diversity_variation,
        output_dir=output_dir,
        filename="creature_diversity_variation.png",
        title="Creature diversity variation per frame",
    )
    plot_num_creatures(
        creature_diversity_cumulative,
        output_dir=output_dir,
        filename="diversity_per_frame.png",
        title="Diversity per frame",
    )
    plot_num_creatures(
        detection_num,
        output_dir=output_dir,
        filename="num_detections.png",
        title="Number of detections per frame",
    )
    plot_num_creatures(
        creature_num,
        output_dir=output_dir,
        filename="num_creatures.png",
        title="Number of creatures per frame",
    )
    plot_num_creatures(
        pokedex_size,
        output_dir=output_dir,
        filename="num_creatures_pokedex.png",
        title="Encyclopdia size per frame",
    )
    plot_encyclopedia(pokedex, output_dir=output_dir, filename="encyclopedia.png")

    print("Simulation duration (frames): ", frame_count)
    print("Max number of tracks", np.max(np.sum(creature_diversity, axis=0)))
    print(
        "Max number of creature_types", np.max(np.sum(creature_diversity > 0, axis=0))
    )
    print("Number of discovered creatures", pokedex.get_num_types())


def parse_args():
    """Parse the arguments for the script"""
    parser = argparse.ArgumentParser(
        description="Biodiversity analysis in 2D Cellular Automata"
    )
    parser.add_argument(
        "--patch_size", type=int, default=15, help="Patch size for creature search"
    )
    parser.add_argument(
        "--max_frames", type=int, default=400, help="Maximum number of frames"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_encyclopedia_rule_search",
        help="Output directory",
    )
    parser.add_argument("--width", type=int, default=400, help="Width of the window")
    parser.add_argument("--height", type=int, default=300, help="Height of the window")
    parser.add_argument(
        "--fps", type=int, default=30, help="Visualization (target) frames per second"
    )
    parser.add_argument(
        "--recording", action="store_true", help="Boolean for recording"
    )
    parser.add_argument(
        "--non_launch_vid", action="store_true", help="Boolean for launching video"
    )
    parser.add_argument(
        "--non_random", action="store_true", help="Use random initial state"
    )
    parser.add_argument(
        "--classify", action="store_true", help="classify the creatures"
    )
    parser.add_argument(
        "--plot_detection",
        action="store_true",
        help="Plot the detection during simulation",
    )
    parser.add_argument(
        "--plot_creatures",
        action="store_true",
        help="Plot the detection during simulation",
    )
    parser.add_argument(
        "--dec_mode",
        default="Kmeans",
        help="Method to classify the creatures (PCA, ICA, Kmeans)",
    )
    parser.add_argument(
        "--b_rule",
        type=str,
        default="3",
        help="Rule for the cellular automaton (B/S notation)",
    )
    parser.add_argument(
        "--s_rule",
        type=str,
        default="23",
        help="Rule for the cellular automaton (B/S notation)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the simulation without window display",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    W = args.width
    H = args.height
    fps = args.fps
    recording = args.recording
    launch_vid = not args.non_launch_vid
    patch_size = args.patch_size
    max_frames = args.max_frames
    random = not args.non_random
    classify = args.classify
    dec_mode = args.dec_mode
    b_rule = args.b_rule
    s_rule = args.s_rule
    headless = args.headless

    run_simulation(
        W,
        H,
        fps,
        recording,
        launch_vid,
        patch_size,
        max_frames,
        random,
        classify,
        dec_mode,
        b_rule,
        s_rule,
        headless,
    )
