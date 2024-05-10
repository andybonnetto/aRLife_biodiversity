import numpy as np
import os
import matplotlib.pyplot as plt


class Creature:
    def __init__(
        self,
        creature_data,
        creature_type: int = None,
        num_creat=1,
        creature_pos: list = None,
    ):
        """Creature class to store the data of a creature and its type"""
        self.creature_data = creature_data
        self.creature_type = creature_type
        self.num_creat = num_creat
        self.x = creature_pos[0]
        self.y = creature_pos[1]

    def __repr__(self):
        """Representation of the creature object"""
        return f"Creature of type {self.creature_type}"

    def __str__(self):
        """String representation of the creature object"""
        return f"Creature of type {self.creature_type}"

    def get_rectangle(self, patch_size):
        return self.x, self.y, patch_size, patch_size


class Encyclopedia:
    """Class to store the creatures and their types in a dictionary and a numpy array of creatures data"""

    def __init__(self, patch_size, zoom_ratio=2):
        """Initialize the encyclopedia with an empty dictionary and a numpy array of zeros of size 1xpatch_size*zoom_ratio x patch_size*zoom_ratio"""
        self.creatures = []
        self.tracks = []
        self.encyclopedia_page = np.zeros(
            (1, patch_size * zoom_ratio, patch_size * zoom_ratio)
        )
        self.patch_size = patch_size

    def _check_is_in(self, creature: Creature):
        if len(self.creatures) > 0:
            comp = (
                np.sum(self.encyclopedia_page - creature.creature_data, axis=(1, 2))
                == 0
            )
            if np.sum(comp):
                return True, np.where(comp)[0][0]
        return False, None

    def _check_overlap(self, x1, y1, x2, y2, margin=1):
        if (
            x1 < x2 + self.patch_size + margin
            and x1 + self.patch_size + margin > x2
            and y1 < y2 + self.patch_size + margin
            and y1 + self.patch_size + margin > y2
        ):
            return True
        return False

    def update(self, creature: Creature):
        if np.sum(creature.creature_data):  # Check if there is a creature
            check, c_id = self._check_is_in(creature)
            if not check:  # Check whether the creature is already in the encyclopedia
                self.creatures.append(creature)
                self.tracks.append([[creature.x, creature.y, 0, 0]]) #0 is the age of the track
                self.encyclopedia_page = np.concatenate(
                    [self.encyclopedia_page, creature.creature_data[np.newaxis, :, :]],
                    axis=0,
                )
            else:
                overlaps = []
                for track in self.tracks[c_id - 1]:
                    overlaps.append(
                        self._check_overlap(track[0], track[1], creature.x, creature.y)
                    )
                if not np.sum(overlaps):
                    self.creatures[
                        c_id - 1
                    ].num_creat += 1  # Increment creature count only if the creature is not overlapping with existing creature
                    self.tracks[c_id - 1].append([creature.x, creature.y, 0, 0])
                    self.creatures[c_id - 1].x = creature.x
                    self.creatures[c_id - 1].y = creature.y
                else:
                    ind = np.where(overlaps)[0][0]
                    self.tracks[c_id - 1][ind] = [creature.x, creature.y, 0, self.tracks[c_id - 1][ind][3] + 1]
    def save(self, output_dir:str):
        # Save the encyclopedia
        np.save(os.path.join(output_dir, "encyclopedia.npy"), self.encyclopedia_page)
    
    def save_creatures(self, output_dir: str):
        

        # Save images of the creatures
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        for c_id, creature in enumerate(self.creatures):
            f = plt.figure(figsize=(10, 10))
            plt.imshow(
                creature.creature_data,
                aspect="auto",
                interpolation="nearest",
                cmap="cubehelix",
            )
            plt.grid(False)
            plt.axis("off")
            im_dir = os.path.join(output_dir, "images", str(creature.creature_type))
            os.makedirs(im_dir, exist_ok=True)
            plt.savefig(
                os.path.join(im_dir, f"creature_{c_id}_{creature.creature_type}.png")
            )
            plt.close()
            
    def update_tracks(self):
        for t in range(len(self.tracks)):
            if len(self.tracks[t]):
                tracks = np.array(self.tracks[t])
                tracks[:,2] += 1
                tracks[:,3] += 1
                tracks = tracks[np.where(tracks[:,2] <= 2)[0]]
                self.tracks[t] = tracks.tolist()
            

    def get_num_creatures(self):
        """Return the number of creatures in the encyclopedia"""
        return np.sum(creat.num_creat for creat in self.creatures)
    
    def get_num_creatures_per_type(self):
        '''Return the number of creatures per type in the encyclopedia'''
        return np.array([creat.num_creat for creat in self.creatures])

    def get_num_types(self):
        """Return the number of types of creatures in the encyclopedia"""
        return len(self.creatures)
    
    def get_num_tracks(self):
        """Return the number of tracks in the encyclopedia"""
        return np.sum([len(track) for track in self.tracks])
    
    def get_track_ages(self):
        '''Return the cumulative ages of the tracks in the encyclopedia'''
        if len(self.tracks):
            return np.sum([np.sum(np.array(track)[:,3]) for track in self.tracks if len(track)])
        else:
            return 0
