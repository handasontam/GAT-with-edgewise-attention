#standardSQL

WITH two_hob_trans AS (
  WITH one_hob_address AS (
    WITH temp_one_hob_address AS(
      SELECT
        trace.from_address AS address
      FROM
        `bigquery-public-data.ethereum_blockchain.traces` AS trace
      JOIN
        `ethereum-data-mining.ExtractedFeatures.Labels` AS labels
      ON
        (labels.address = trace.to_address)
      WHERE 
        block_timestamp < TIMESTAMP('2018-12-02')
        AND block_timestamp > TIMESTAMP('2017-12-02')
        AND status = 1
        AND (call_type NOT IN ('delegatecall',
            'callcode',
            'staticcall')
          OR call_type IS NULL)
      UNION ALL
      SELECT
        trace.to_address AS address
      FROM
        `bigquery-public-data.ethereum_blockchain.traces` AS trace
      JOIN
        `ethereum-data-mining.ExtractedFeatures.Labels` AS labels
      ON
        (labels.address = trace.from_address)
      WHERE 
        block_timestamp < TIMESTAMP('2018-12-02')
        AND block_timestamp > TIMESTAMP('2017-12-02')
        AND status = 1
        AND (call_type NOT IN ('delegatecall',
            'callcode',
            'staticcall')
          OR call_type IS NULL)
      UNION ALL
      -- miner address (from seller to miner)
      SELECT
        blocks.miner AS address
      FROM
        `bigquery-public-data.ethereum_blockchain.transactions` AS transactions
      JOIN
        `bigquery-public-data.ethereum_blockchain.blocks` AS blocks
      ON
        blocks.number = transactions.block_number
      JOIN
        `ethereum-data-mining.ExtractedFeatures.Labels` AS labels
      ON
        (labels.address = transactions.from_address)
      WHERE block_timestamp < TIMESTAMP('2018-12-02')
        AND block_timestamp > TIMESTAMP('2017-12-02')
      UNION ALL
      -- seller address (from seller to miner)
      SELECT
        transactions.from_address AS address
      FROM
        `bigquery-public-data.ethereum_blockchain.transactions` AS transactions
      JOIN
        `bigquery-public-data.ethereum_blockchain.blocks` AS blocks
      ON
        blocks.number = transactions.block_number
      JOIN
        `ethereum-data-mining.ExtractedFeatures.Labels` AS labels
      ON
        (labels.address = blocks.miner)
      WHERE block_timestamp < TIMESTAMP('2018-12-02')
        AND block_timestamp > TIMESTAMP('2017-12-02')
      -- the original set of nodes
      UNION ALL
      SELECT
        labels.address AS address
      FROM
        `ethereum-data-mining.ExtractedFeatures.Labels` AS labels
        )
    SELECT 
      DISTINCT address
    FROM temp_one_hob_address

  )
    SELECT
      trace.from_address AS from_address, 
      trace.to_address AS to_address, 
      trace.value AS value, 
      trace.call_type AS call_type, 
      trace.block_timestamp AS block_timestamp
    FROM
      `bigquery-public-data.ethereum_blockchain.traces` AS trace
    JOIN
      one_hob_address
    ON
      (one_hob_address.address = trace.to_address
      OR one_hob_address.address = trace.from_address)
    WHERE block_timestamp < TIMESTAMP('2018-12-02')
        AND block_timestamp > TIMESTAMP('2017-12-02')
        AND to_address IS NOT NULL
        AND from_address IS NOT NULL
        AND status = 1
        AND (call_type NOT IN ('delegatecall',
            'callcode',
            'staticcall')
          OR call_type IS NULL)
    UNION ALL
    -- miner address (from seller to miner)
    SELECT
      blocks.miner AS to_address, 
      transactions.from_address AS from_address, 
      CAST(receipt_gas_used AS numeric) * CAST(gas_price AS numeric) AS value, 
      'mining' AS call_type, 
      block_timestamp
    FROM
      `bigquery-public-data.ethereum_blockchain.transactions` AS transactions
    JOIN
      `bigquery-public-data.ethereum_blockchain.blocks` AS blocks
    ON
      blocks.number = transactions.block_number
    JOIN
      one_hob_address
    ON
      (one_hob_address.address = transactions.from_address
      OR one_hob_address.address = blocks.miner)
    WHERE block_timestamp < TIMESTAMP('2018-12-02')
        AND block_timestamp > TIMESTAMP('2017-12-02')
        AND to_address IS NOT NULL
        AND from_address IS NOT NULL
)

-- main query
SELECT 
  from_address, 
  to_address, 
  value, 
  call_type
  -- sum(value/1E17) as sum_value, 
  -- avg(value/1E17) as avg_value, 
  -- variance(value/1E17) as var_value, 
  -- count(*) as count_trans, 
  -- -- SUM(CASE when call_type = 'delegatecall' THEN 1 ELSE 0 END) AS count_delegatecall, 
  -- -- SUM(CASE when call_type = 'callcode' THEN 1 ELSE 0 END) AS count_callcode, 
  -- -- SUM(CASE when call_type = 'staticcall' THEN 1 ELSE 0 END) AS count_staticcall, 
  -- SUM(CASE when call_type = 'call' THEN 1 ELSE 0 END) AS count_call,  
  -- SUM(CASE when call_type = 'mining' THEN 1 ELSE 0 END) AS count_mining,  
  -- SUM(CASE when (call_type != 'delegatecall' AND call_type != 'callcode' AND call_type != 'staticcall' AND call_type != 'mining' AND call_type != 'call') THEN 1 ELSE 0 END) AS count_other
FROM
  two_hob_trans
-- GROUP BY
--   from_address, to_address



