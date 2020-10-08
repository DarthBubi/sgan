import os
import numpy as np
import torch
import rospy

from attrdict import AttrDict
from collections import deque, OrderedDict

from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs

from geometry_msgs.msg import Point
from spencer_tracking_msgs.msg import TrackedPersons, TrackedPerson
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
        self.generator = self.get_generator(checkpoint)
        self.args_ = AttrDict(checkpoint['args'])
        self.tracked_persons_sub = rospy.Subscriber(
            "/spencer/perception/tracked_persons", TrackedPersons, self.tracked_persons_cb, queue_size=3)
        # self.predicted_tracks_pub = rospy.Publisher("/sgan/predictions", TBD, queue_size=1)
        self.predictions_marker_pub = rospy.Publisher("/sgan/predictions_marker", Marker, queue_size=1)
        self.tracked_persons = {}
        self.max_age = rospy.Duration(10)

    def get_generator(self, checkpoint):
        args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
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

        if type(self.tracked_persons) is dict:
            self.tracked_persons = OrderedDict(
                sorted(self.tracked_persons.items(), key=lambda t: t[0]))

        # for key in self.tracked_persons.keys():
        #     if self.tracked_persons[key].:

    def predict_tracks(self):
        """
        docstring
        """

        curr_seq = np.zeros(self.seq_len, )
        for id, tracks, i in enumerate(self.tracked_persons.copy().items()):  # copy is needed, because the subscriber can mutate the dict
            if len(tracks) != self.seq_len:
                continue
            msg = Marker()
            msg.header.frame_id = "odom"
            msg.ns = "track_" + str(id)
            msg.id = 0
            msg.action = msg.ADD
            msg.pose.orientation.w = 1.0
            msg.type = msg.LINE_STRIP
            msg.points = []
            msg.scale.x = 0.05
            msg.color.a = 1.
            msg.color.r = 1.
            msg.color.g = 0.
            msg.color.b = 0.

            with torch.no_grad():
                obs_traj = np.array([[track.pose.pose.position.x, track.pose.pose.position.y] for track in tracks])
                obs_traj_rel = np.zeros(obs_traj.shape)
                obs_traj_rel[1:, :] = obs_traj[1:, :] - obs_traj[:-1, :]
                obs_traj = obs_traj[:, :, np.newaxis]
                obs_traj = obs_traj.reshape(self.seq_len, 1, 2)
                obs_traj_rel = obs_traj_rel[:, :, np.newaxis]
                obs_traj_rel = obs_traj_rel.reshape(obs_traj.shape)
                obs_traj = torch.from_numpy(obs_traj).float().cuda()
                obs_traj_rel = torch.from_numpy(obs_traj_rel).float().cuda()
                rospy.logdebug_throttle(1, str(id) + ": " + str(obs_traj))
                rospy.logdebug_throttle(1, "shape: " + str(obs_traj.shape))
                l = [1]
                cum_start_idx = [0] + np.cumsum(l).tolist()
                seq_start_end = [
                    (start, end)
                    for start, end in zip(cum_start_idx, cum_start_idx[1:])
                ]
                seq_start_end = torch.Tensor(seq_start_end).type(torch.int).cuda()
                msg.points = []
                for _ in range(self.num_samples):
                    pred_traj_fake_rel = self.generator(obs_traj, obs_traj_rel, seq_start_end)
                    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                    msg.points.extend([Point(pred[0][0], pred[0][1], 0) for pred in pred_traj_fake.tolist()])
                self.predictions_marker_pub.publish(msg)

    def spin(self):
        """
        docstring
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.tracked_persons:
                self.predict_tracks()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("sgan_predictor")
    sgan_node = SGANNode()
    sgan_node.spin()
