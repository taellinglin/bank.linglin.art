from panda3d.core import Vec3
from panda3d.bullet import BulletRigidBodyNode, BulletBoxShape, TransformState, BulletWorld, BulletGenericConstraint, BTConstraintParam
from direct.showbase import ShowBase
import random

class LevelGenerator:
    def __init__(self, base):
        self.base = base
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self.objects = []
        
    def create_money_bill(self, pos=(0, 0, 5), size=1.0, thickness=0.005):
        """
        Create a money bill as a thin box with physics
        
        Args:
            pos (tuple): Position (x, y, z)
            size (float): Size of the bill
            thickness (float): Thickness of the bill
        """
        # Create visual model
        bill_model = self.base.loader.loadModel("models/box")
        bill_model.setScale(size, size/2, thickness)  # Rectangular bill shape
        bill_model.setPos(pos)
        bill_model.setColor(0.2, 0.6, 0.2, 1)  # Green money color
        bill_model.reparentTo(self.base.render)
        
        # Create physics shape
        shape = BulletBoxShape(Vec3(size/2, size/4, thickness/2))
        
        # Create rigid body
        body = BulletRigidBodyNode('money_bill')
        body.setMass(0.1)  # Light weight like paper
        body.addShape(shape)
        body.setFriction(0.3)
        body.setRestitution(0.1)  # Low bounciness
        body.setDamping(0.8, 0.8)  # High damping for paper-like movement
        
        # Set initial position
        body.setTransform(TransformState.makePos(pos))
        
        # Add to world and attach to model
        bill_np = self.base.render.attachNewNode(body)
        self.world.attachRigidBody(body)
        bill_model.reparentTo(bill_np)
        
        self.objects.append(bill_np)
        return bill_np
        
    def create_ground(self, size=20, height=0.5, pos=(0, 0, 0)):
        """
        Create a ground plane for collision
        
        Args:
            size (float): Size of the ground
            height (float): Height/thickness of the ground
            pos (tuple): Position (x, y, z)
        """
        # Create visual
        ground_model = self.base.loader.loadModel("models/box")
        ground_model.setScale(size, size, height)
        ground_model.setPos(pos)
        ground_model.setColor(0.4, 0.3, 0.2, 1)  # Brown ground color
        ground_model.reparentTo(self.base.render)
        
        # Create physics shape
        shape = BulletBoxShape(Vec3(size, size, height))
        
        # Create static body
        body = BulletRigidBodyNode('ground')
        body.setMass(0)  # Static object
        body.addShape(shape)
        body.setFriction(0.8)
        
        # Set position
        body.setTransform(TransformState.makePos(pos))
        
        # Add to world
        ground_np = self.base.render.attachNewNode(body)
        self.world.attachRigidBody(body)
        ground_model.reparentTo(ground_np)
        
        self.objects.append(ground_np)
        return ground_np
        
    def create_wind_force(self, strength=5.0, pos=(0, 0, 3), direction=(1, 0, 0)):
        """
        Create a wind force field
        
        Args:
            strength (float): Strength of the wind
            pos (tuple): Position of the wind source
            direction (tuple): Direction vector of the wind
        """
        force = BulletGenericConstraint(self.objects[0].node(), TransformState.makePos(pos))
        force.setLinearLowerLimit(Vec3(0, 0, 0))
        force.setLinearUpperLimit(Vec3(0, 0, 0))
        force.setAngularLowerLimit(Vec3(0, 0, 0))
        force.setAngularUpperLimit(Vec3(0, 0, 0))
        
        # Apply constant force in direction
        force.setParam(BTConstraintParam.ERP, 0.8)
        force.setParam(BTConstraintParam.CFM, 0.1)
        
        self.world.attachConstraint(force)
        return force
        
    def create_money_stack(self, count=5, base_pos=(0, 0, 2), spacing=0.01):
        """
        Create a stack of money bills
        
        Args:
            count (int): Number of bills in stack
            base_pos (tuple): Base position of the stack
            spacing (float): Spacing between bills
        """
        bills = []
        for i in range(count):
            pos = (base_pos[0], base_pos[1], base_pos[2] + i * spacing)
            bill = self.create_money_bill(pos=pos, size=0.8)
            bills.append(bill)
        return bills
        
    def apply_random_force(self, bill_np, min_force=2, max_force=8):
        """
        Apply a random force to simulate natural movement
        
        Args:
            bill_np: The bill node path
            min_force (float): Minimum force magnitude
            max_force (float): Maximum force magnitude
        """
        force = Vec3(
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(0, 0.5)
        )
        force.normalize()
        force *= random.uniform(min_force, max_force)
        
        bill_np.node().applyCentralForce(force)
        
    def check_ground_collision(self, bill_np, ground_np, threshold=0.1):
        """
        Check if a bill is resting on the ground
        
        Args:
            bill_np: The bill node path
            ground_np: The ground node path
            threshold (float): Distance threshold for ground contact
        """
        bill_pos = bill_np.getPos()
        ground_pos = ground_np.getPos()
        ground_height = ground_pos.z + ground_np.getScale().z
        
        return bill_pos.z - threshold <= ground_height
        
    def update_physics(self, dt):
        """Update the physics simulation"""
        self.world.doPhysics(dt)
        
    def cleanup(self):
        """Clean up all physics objects"""
        for obj in self.objects:
            if obj.node():
                self.world.removeRigidBody(obj.node())
        self.objects = []

# Example usage class
class MoneySimulationDemo(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Setup level generator
        self.level_gen = LevelGenerator(self)
        
        # Create ground
        self.ground = self.level_gen.create_ground()
        
        # Create money bills
        self.money_bills = [
            self.level_gen.create_money_bill(pos=(0, 0, 8)),
            self.level_gen.create_money_bill(pos=(2, 0, 6)),
            self.level_gen.create_money_bill(pos=(-2, 0, 7))
        ]
        
        # Create a stack of money
        self.money_stack = self.level_gen.create_money_stack(count=3, base_pos=(3, 0, 1))
        
        # Setup camera
        self.disableMouse()
        self.camera.setPos(0, -10, 5)
        self.camera.lookAt(0, 0, 0)
        
        # Update task
        self.taskMgr.add(self.update, 'update')
        
    def update(self, task):
        dt = globalClock.getDt()
        self.level_gen.update_physics(dt)
        
        # Occasionally apply random forces to bills
        if random.random() < 0.02:  # 2% chance per frame
            bill = random.choice(self.money_bills)
            self.level_gen.apply_random_force(bill)
            
        return task.cont

# To run the demo
if __name__ == "__main__":
    demo = MoneySimulationDemo()
    demo.run()