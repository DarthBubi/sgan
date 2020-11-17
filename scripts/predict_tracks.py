import os
import numpy as np
import torch
import rospy

from attrdict import AttrDict
from collections import deque, OrderedDict
from copy import copy

from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs

from geometry_msgs.msg import Point
from spencer_tracking_msgs.msg import TrackedPersons
from visualization_msgs.msg import Marker, MarkerArray


class SGANNode(object):
    """
    docstring
    """

    def __init__(self):
        model_path = rospy.get_param("model_path")
        model_name = rospy.get_param("model_name")
        model = os.path.join(model_path, model_name + "_model.pt")
        checkpoint = torch.load(model)
        self.num_samples = rospy.get_param("num_samples", 20)
        self.seq_len = rospy.get_param("seq_len", 8)
        self.visualise = rospy.get_param("visualise", True)
        self.args_ = AttrDict(checkpoint['args'])
        self.generator = self.get_generator(checkpoint)
        self.tracked_persons_sub = rospy.Subscriber(
            "/spencer/perception/tracked_persons", TrackedPersons, self.tracked_persons_cb, queue_size=3)
        # self.predicted_tracks_pub = rospy.Publisher("/sgan/predictions", TBD, queue_size=1)
        self.predictions_marker_pub = rospy.Publisher("/sgan/predictions_marker", MarkerArray, queue_size=1)
        self.tracked_persons = {}

    def get_generator(self, checkpoint):
        #args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=self.args_.obs_len,
            pred_len=self.args_.pred_len,
            embedding_dim=self.args_.embedding_dim,
            encoder_h_dim=self.args_.encoder_h_dim_g,
            decoder_h_dim=self.args_.decoder_h_dim_g,
            mlp_dim=self.args_.mlp_dim,
            num_layers=self.args_.num_layers,
            noise_dim=self.args_.noise_dim,
            noise_type=self.args_.noise_type,
            noise_mix_type=self.args_.noise_mix_type,
            pooling_type=self.args_.pooling_type,
            pool_every_timestep=self.args_.pool_every_timestep,
            dropout=self.args_.dropout,
            bottleneck_dim=self.args_.bottleneck_dim,
            neighborhood_size=self.args_.neighborhood_size,
            grid_size=self.args_.grid_size,
            batch_norm=self.args_.batch_norm,
            use_gpu=self.args_.use_gpu)
        generator.load_state_dict(checkpoint['g_state'])
        if self.args_.use_gpu:
            generator.cuda()
        generator.train()
        return generator

    def tracked_persons_cb(self, msg):
        """
        docstring
        """
        for track in msg.tracks:
            if track.track_id in self.tracked_persons:
                self.tracked_persons[track.track_id].append(track)
            else:
                self.tracked_persons[track.track_id] = deque([track], maxlen=8)

        current_ids = [track.track_id for track in msg.tracks]
        msg_arr = MarkerArray()
        for k in self.tracked_persons.copy().keys():
            if k not in current_ids:
                if self.visualise:
                    msg = Marker()
                    msg.action = msg.DELETEALL
                    msg.ns = "track_" + str(k)
                    msg_arr.markers.append(copy(msg))

                del self.tracked_persons[k]

        if self.visualise:
            self.predictions_marker_pub.publish(msg_arr)

        if type(self.tracked_persons) is dict:
            self.tracked_persons = OrderedDict(
                sorted(self.tracked_persons.items(), key=lambda t: t[0]))

    def predict_tracks(self):
        """
        docstring
        """
        curr_seq = np.zeros((self.seq_len, len(self.tracked_persons), 2))
        curr_seq_rel = np.zeros(curr_seq.shape)
        valid_tracks = 0

        with torch.no_grad():
            # copy is needed, because the subscriber callback can mutate the dict
            for id, tracks in self.tracked_persons.copy().items():
                if len(tracks) != self.seq_len:
                    continue
                obs_traj = np.array([[track.pose.pose.position.x, track.pose.pose.position.y] for track in tracks])
                obs_traj_rel = np.zeros(obs_traj.shape)
                obs_traj_rel[1:, :] = obs_traj[1:, :] - obs_traj[:-1, :]
                curr_seq[:, valid_tracks, :] = obs_traj
                curr_seq_rel[:, valid_tracks, :] = obs_traj_rel
                valid_tracks = valid_tracks + 1

            if valid_tracks > 0:
                curr_seq = torch.from_numpy(curr_seq[:, :valid_tracks, :]).float()
                curr_seq_rel = torch.from_numpy(curr_seq_rel[:, :valid_tracks, :]).float()
                if self.args_.use_gpu:
                    curr_seq = curr_seq.cuda()
                    curr_seq_rel = curr_seq_rel.cuda()
                l = [valid_tracks]
                cum_start_idx = [0] + np.cumsum(l).tolist()
                seq_start_end = [
                    (start, end)
                    for start, end in zip(cum_start_idx, cum_start_idx[1:])
                ]
                seq_start_end = torch.Tensor(seq_start_end).type(torch.int)
                if self.args_.use_gpu:
                    seq_start_end = seq_start_end.cuda()
                pred_samples = dict.fromkeys(self.tracked_persons.keys(), [])
                track_ids = list(self.tracked_persons.keys())

                for _ in range(self.num_samples):
                    pred_traj_fake_rel = self.generator(curr_seq, curr_seq_rel, seq_start_end)
                    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, curr_seq[-1])
                    for ped_id, pred in enumerate(pred_traj_fake.transpose(1, 0).tolist()):
                        pred_samples[track_ids[ped_id]].extend(
                            [Point(pred[idx][0], pred[idx][1], 0) for idx in range(0, self.seq_len)])

                if self.visualise:
                    self.visualise_predictions(pred_samples)

    def visualise_predictions(self, predictions_samples):
        msg = Marker()
        msg.header.frame_id = "odom"
        msg.ns = "tracks"
        msg.id = 0
        msg.action = msg.ADD
        msg.pose.orientation.w = 1.0
        msg.type = msg.LINE_LIST
        msg.points = []
        msg.scale.x = 0.05
        msg.color.a = 1.
        msg.color.r = 1.
        msg.color.g = 0.
        msg.color.b = 0.

        msg_arr = MarkerArray()

        for k, v in predictions_samples.items():
            msg.points = []
            msg.points.extend(v)
            msg.ns = "track_" + str(k)
            msg_arr.markers.append(copy(msg))

        self.predictions_marker_pub.publish(msg_arr)

    def spin(self):
        """
        docstring
        """
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.tracked_persons:
                self.predict_tracks()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("sgan_predictor")
    sgan_node = SGANNode()
    sgan_node.spin()
