# C++ example using Routing APIs

This C++ example demonstrates using Routing APIs to construct a
data model and pass it to the cuOpt solver for optimization and
get back the routing solution.


There are three examples included:
1. `cvrp_daily_deliveries.cu`
Micro fulfillment centers allow retailers to move predictable, high volume products closer to the end consumer allowing for lower costs and shorter overall delivery times.

In this scenario we have a number of same-day delivery orders that we would like to process for a given area from a given micro fulfillment center. We have the requisite number of delivery vehicles and enough time to deliver all packages over the course of a single day. Each delivery vehicle has a maximum capacity of orders it can carry and we are looking for the route assignment that minimizes the total distance driven by all vehicles.
2. `pdptw_mixed_fleet.cu`
In scenarios such as food delivery, the delivery fleet may consist of various types of vehicles, for example bikes, and cars, and each type of vehicle has own advantages and limitations. For example, in crowded streets of NYC, it might be faster to reach a nearby destination on bike compared to car, while it is much faster with car in suburban areas. Service provides can improve customer satisfaction, reduce costs, and increase earning opportunity for drivers, using various types of vehicles depending on the geography of the service area.
3. `service_team_routing.cpp`
The ability of service providers to set service time windows allows for easier and more dependable coordination between the service provider and their customers, while increasing overall customer satisfaction.

In this scenario we have a number of service order locations with associated time windows and service times (time on-site to complete service). Each technician has an associated availability, ability to complete certain types of service, and a maximum number of service appointments per day.

## Compile and execute

To configure, compile, and execute the tests in the current working directory, follow these steps:

```bash
# Configure, build and execute the tests
cmake -S . -B build/; cmake --build build/ --parallel $PARALLEL_LEVEL; ./service_team_routing; ./pdptw_mixed_fleet; ./cvrp_daily_deliveries
```
