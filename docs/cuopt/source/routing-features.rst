========================
Routing Features
========================

Availability
------------------------------

The Routing solver is available in two forms:

- **As a Self-Hosted Service**: cuOpt Routing can be deployed as a service in your own infrastructure, enabling you to maintain full control while integrating it into your existing systems. For more information, see :doc:`cuopt-server/quick-start`.

- **Python SDK**: A Python package that provides direct access to cuOpt's routing capabilities through a simple, intuitive API. This allows for seamless integration into Python applications and workflows. For more information, see :doc:`cuopt-python/quick-start`.

Both options provide the same routing optimization capabilities while offering flexibility in deployment and integration approaches.

Heterogeneous Fleet
------------------------------

In a Vehicle Routing Problem (VRP), enterprise customers may have a fleet that is composed of different vehicles, such as trucks and motorbikes. Each vehicle type will likely have different costs, travel times, different constraints such as time windows, start and end locations, and carrying capacity.  To model such scenarios, users can provide multiple input matrices (travel time matrix/cost), one for each type.  Constraints must be provided for each vehicle.

Multiple Input Matrices
------------------------------

A cost matrix needs to be a square matrix of some travel metric that is passed to the NVIDIA cuOpt solver. In many variations of the Traveling Salesman Problem (TSP), this includes travel time, distance, or another business cost. A cost matrix is a square matrix that represents the cost of traveling between each two pairs of locations in the problem. ``cost matrix`` can hold any cost matrix; the cost can be travel time, travel distance, dollar cost, or any weighted function of cost. If the cost matrix doesn't use travel time, ``travel time matrix`` should be used so that the cost matrix is used for optimization, while the travel time matrix is used for meeting time constraints.

In the case of a heterogeneous fleet, different vehicle types may have different travel times. To support this, multiple ``cost matrix`` and ``travel time matrix`` can be provided such that each vehicle type has a corresponding matrix.

Multiple Input Waypoint Graphs
------------------------------

Similar to cost matrix, distance and time details can be provided through waypoint graphs through ``cost waypoint graph`` or ``travel time waypoint graph``. Each vehicle type will have its own waypoint graph data for both cost and time.

Vehicle Time Windows
------------------------------

Time windows represent the operating time of the vehicles. There is one time window per vehicle. This data is represented as an integer. Raw data may include Universal Time Stamp (UTC) date/time format or string format which must be converted to floating value. (Example: 9:00 am - 6:00 pm converted to minutes in a 24-hour period starting at 12:00 am, would be ``[540, 1080]``). All time/cost units provided to the cuOpt solver should be in the same unit. If the travel time cost matrix includes the travel time in between locations in minutes, the time windows must be integer representation of minutes as well.

.. note::

    All time windows are expected to be => 0, and all time windows are int32 types. So the time window range supported would be ``[0, 2^31-1]``; this translates to ``[January 1, 1970, 00:00:00, January 19, 2038, 03:14:07]`` in UTC.


Vehicle Breaks
------------------------------

In addition to vehicle time windows, vehicles may also have break time windows. If the time windows represent a driver's working hours in a day, the break time window may represent their lunch break in the middle of the day. The integer representation is consistent with the time windows. In case raw input data is in UTC timestamp, if a vehicle is working a shift of 9:00 am - 6:00 pm, then in a 24-hour period this is equivalent to ``540 - 1080``. If the vehicle has a break from 1:00 pm - 1:30 pm, then the break time window would be ``[780, 810]``.

Along with time, users can also specify possible break locations using ``vehicle break locations``.

There are two types of breaks,

- Uniform breaks - Where each driver in the fleet has same number breaks that they can avail. Like 3 breaks in a shift (coffee, lunch, coffee). This is achieved through ``vehicle break time windows`` and ``vehicle break durations``.

- Non-Uniform breaks - Where each driver might need to take different set of breaks, may be someone called-in sick or some set of workers need more breaks since they work extra hours. This is done through ``vehicle breaks``.

Only one of the type of breaks can be used at a time.


Prize Collection
------------------------
The *prizes* may be used to set a prize for each task. This supports cases where all tasks are not feasible or where-in servicing all tasks are not as profitable as dropping some low prize tasks and servicing the rest and that the user requires tasks with the highest prize to be considered before trying other tasks. If the cost of serving a customer is too high compared to the prize of serving a customer, the solver will drop that customer to improve overall cost.

If the primary goal is to serve as many customers as possible, it is advisable to set very high prizes. Prize collection would be an effective way to drop infeasible tasks and prioritizing tasks with higher prizes. The main difference is in objective function, the total prize will be negated rather than getting added to overall cost. If the objective weight corresponding to prize objective is set to zero, the prize collection mechanism is disabled.

Objectives
------------------------------

The default objective of the solver is to optimize the cost computed using cost matrices.  However, a few other objectives can be set. The final objective is the linear combination of all the objectives specified. The specified objective weights are used as coefficients.  By default, the cuOpt solver optimizes for vehicle count first and final objective next.

Drop First / Last Trips
------------------------------

By dropping the first or last trip, the cuOpt solver does not take into account the vehicles' trip from their depots to their first stop, or the trip from their last stop back to the depots. With these parameters, the route includes only travel costs and times between task locations. In cases where drivers may start their shift from their home location instead of an assigned depot, the trip to and from the depot is unnecessary.

Pickup and Deliveries
------------------------------

Some use cases may include picking up an order from one location and delivering it to another. Each order has two corresponding locations: one for pickup and one for delivery. The same vehicle must handle both the pickup and delivery of the same order, and the pickup of the order must occur prior to the delivery.

Precise Time Limits
------------------------------

It is recommended to use the default time limit (which is ``10 + number_of_nodes/6``) seconds. If the time limit is set to X seconds, then solver continues to run for X seconds even when there is no improvement in the solution quality.

.. note::

    In case of self-hosted version, the ``time_limit`` set is what solver will essentially use to solve the problem and doesn't include ``network transfer``, ``etl``, ``validation of input``, ``instance being busy with other requests`` and few other overheads, these overheads would be comparatively smaller. So the overall request to response round trip time would be ``solve_time`` + ``overhead``.

Vehicle Start and End Locations
-----------------------------------------

Each vehicle in the fleet must have a start and end location for the given set of locations in a waypoint graph or cost matrix. These locations must be included in the cost matrices or waypoint graphs. In many use cases, these start and end locations are depots or distribution centers, such that vehicles depart their assigned depot in the morning, fulfill all of their assigned tasks, and then return to the assigned depot at the end of the work day. The start and end locations are not necessarily the same location (e.g., a vehicle departs from depot 1 in the morning but returns to depot 2 at night).

Minimum Constraint on Number of Vehicles
-----------------------------------------

By default, cuOpt tries to minimize the number of vehicles used in the solution and considers the fleet size an upper bound. If a given input fleet has 20 vehicles available but only 10 are needed to fulfill all of the tasks, then the cuOpt solution will include only 10 vehicles. To set a lower bound on the number of vehicles used, set ``min_vehicles`` in ``fleet_data``. If the exact number of vehicles to be used is known, specifying a fleet of the desired size and setting ``min vehicles`` equal to the fleet size will guarantee that all vehicles are used.

Maximum Constraints per Vehicle
-----------------------------------------

Vehicles may have a constraint for maximum distance each vehicle can travel or maximum time a vehicle can operate. This means that even if a vehicle has a time window of 9am to 9pm, and a driver may be available to work for those 12 hours, we can add a constraint that a work day must not exceed 8 of those 12 hours.

Fixed Cost per Vehicle
-----------------------
Vehicles can have different fixed costs associated with them. This helps in scenarios where a single vehicle with a higher cost can be avoided if it can be done with two or more vehicles with lesser costs. This would be dependent on the objective function.

Mapping Orders to Vehicles, and Vehicles to Orders
---------------------------------------------------
By default, cuOpt will assign orders to vehicles based on the optimal routes. However, in some cases, it makes sense to assign specific orders to specific vehicles, or, conversely, specific vehicles to specific orders.

-  ``order vehicle match`` allows assigning orders to vehicles. For example, a food distribution center wants to make shipments to grocery stores around the city. Say the fleet consists of refrigerated trucks, such that they can carry frozen food, and vans, which cannot. In this case, we want to assign the orders that contain frozen food to the trucks (rather than just any vehicle).

-  ``vehicle order match`` allows assigning vehicles to orders. For example, a maintenance company can have many employees (technicians) who can fulfil various tasks. When a customer requests a service, the company may dispatch any available technician to fulfill their request. However, if a customers request a service that only one technician can fulfill, those orders can be assigned to this one technician.

In cases where a set of orders need to be assigned to a set of vehicles, either constraint can be used as long as the mapping is done correctly.

Initial Solution
----------------

Previously run solutions or uploaded solutions can be used as an initial solution to start a new request to boost the speed to the solution. :ref:`Examples <initial-solution-in-python>` are shared on the self-hosted page.
