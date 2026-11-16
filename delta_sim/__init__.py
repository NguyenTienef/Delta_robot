"""Educational Delta robot simulator; distances in mm, angles in degrees."""

from .kinematics import DeltaRobot, Geometry, KinematicsError

__all__ = ['DeltaRobot', 'Geometry', 'KinematicsError']
