# How to make xml

## `xml_stats.py`

### Arguments

- `--xml/-f`: the source xml

### Features

Type 's' to start statistics, 'n' to generate subset config with route number limit, 'r' to generate subset config with road number limit, or any other key to exit.

## `xml_filter.py`

### Arguments

- `--config/-f`: the config file

### Features

Output the xml configured by config.yaml. The balancing logic is:

1. **Count routes per scenario type**
   Build a mapping like `{scenario_type → [routes]}` and get how many routes each scenario currently has.

2. **Compute a “lower bound” (target count)**

   * `min_count` = smallest number of routes among all scenarios
   * `total_roads` = number of distinct (town, road_id) pairs
   * Default lower bound = `max(min_count, total_roads)`
   * The user can override this interactively.
   * Final `target_count` = `max(lower_bound, min_count, total_roads)`

   → This ensures that every scenario keeps at least as many routes as the rarest scenario or as many as there are roads, whichever is larger.

3. **Iteratively remove routes**
   For any scenario with more than `target_count` routes:

   * Group its routes by road
   * Find the road with the most routes of that scenario
   * Remove one route from that road (preferring the one with the most scenarios inside it)
   * Repeat until all scenario types have ≤ `target_count` routes.

4. **Save the balanced result**
   Write the remaining routes back into the same XML file (`output_xml`).

## `xml_supl.py`

### Arguments

- `--xml/-f`: the source xml

### Features

Repeat the routes which have smaller amount to make all route counts = max count.


## `xml_spliter.py`

Split the xml in `n` to adapt `n` workers

### Arguments

- `--xml/-f`: the source xml
  
- `--number/-n`: number of parts to split into