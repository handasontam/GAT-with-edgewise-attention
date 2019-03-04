#standardSQL
WITH
  -- all addresses
  address_book AS (
  SELECT
    address, 
    -- circular resultant length R (only with sample set larger than certain size)
    -- https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Circular_Data_Analysis.pdf
    case when count(1) > 48 then sqrt(pow(sum(cos(extract(hour from ts)/12*acos(-1))), 2)+pow(sum(sin(extract(hour from ts)/12*acos(-1))), 2))/count(1) else null end as R_active_hour,
    count(distinct date(ts)) as active_days
  FROM (
    SELECT
      from_address AS address,
      block_timestamp AS ts
    FROM
      `bigquery-public-data.ethereum_blockchain.traces`
    UNION ALL
    SELECT
      to_address AS address,
      block_timestamp AS ts
    FROM
      `bigquery-public-data.ethereum_blockchain.traces`)
  WHERE
    ts < TIMESTAMP('2018-12-01')
  GROUP BY
    address ),
  -- balance
  balance_book AS (
  WITH
    value_records AS (
      -- debits
    SELECT
      to_address AS address,
      value AS value,
      block_timestamp AS ts
    FROM
      `bigquery-public-data.ethereum_blockchain.traces`
    WHERE
      to_address IS NOT NULL
      AND status = 1
      AND (call_type NOT IN ('delegatecall',
          'callcode',
          'staticcall')
        OR call_type IS NULL)
    UNION ALL
      -- credits
    SELECT
      from_address AS address,
      -value AS value,
      block_timestamp AS ts
    FROM
      `bigquery-public-data.ethereum_blockchain.traces`
    WHERE
      from_address IS NOT NULL
      AND status = 1
      AND (call_type NOT IN ('delegatecall',
          'callcode',
          'staticcall')
        OR call_type IS NULL)
    UNION ALL
      -- transaction fees debits
    SELECT
      miner AS address,
      CAST(receipt_gas_used AS numeric) * CAST(gas_price AS numeric) AS value,
      block_timestamp AS ts
    FROM
      `bigquery-public-data.ethereum_blockchain.transactions` AS transactions
    JOIN
      `bigquery-public-data.ethereum_blockchain.blocks` AS blocks
    ON
      blocks.number = transactions.block_number
    UNION ALL
      -- transaction fees credits
    SELECT
      from_address AS address,
      -(CAST(receipt_gas_used AS numeric) * CAST(gas_price AS numeric)) AS value,
      block_timestamp AS ts
    FROM
      `bigquery-public-data.ethereum_blockchain.transactions` )
  SELECT
    address,
    SUM(value) AS balance
  FROM
    value_records
  WHERE
    ts < TIMESTAMP('2018-12-01')
  GROUP BY
    address ),
  -- token transfer in count and accumulative token types
  token_in_book AS (
    SELECT
      to_address AS address,
      count(1) as token_in_tnx,
      count(distinct token_address) as token_in_type,
      count(distinct from_address) as token_from_addr
    FROM
      `bigquery-public-data.ethereum_blockchain.token_transfers`
  WHERE
    block_timestamp < TIMESTAMP('2018-12-01')
  GROUP BY
    address),
      -- token transfer out count and accumulative token types
  token_out_book AS (
    SELECT
      from_address AS address,
      count(1) as token_out_tnx,
      count(distinct token_address) as token_out_type,
      count(distinct to_address) as token_to_addr
    FROM
      `bigquery-public-data.ethereum_blockchain.token_transfers`
  WHERE
    block_timestamp < TIMESTAMP('2018-12-01')
  GROUP BY
    address),
  -- Out count
  out_book AS (
  SELECT
    from_address AS address,
    COUNT(to_address) AS out_trace_count,
    COUNT(DISTINCT to_address) AS out_addr_count,
    countif(value > 0) as out_transfer_count,
    avg(nullif(value, 0)) as out_avg_amount
  FROM
    `bigquery-public-data.ethereum_blockchain.traces`
  WHERE
    from_address IS NOT NULL
    AND status = 1
    AND block_timestamp < TIMESTAMP('2018-12-01')
  GROUP BY
    address ),
  -- In count
  in_book AS (
  SELECT
    to_address AS address,
    COUNT(from_address) AS in_trace_count,
    COUNT(DISTINCT from_address) AS in_addr_count,
    countif(value > 0) as in_transfer_count,
    avg(nullif(value, 0)) as in_avg_amount,
    
    avg(if(trace_type="call", nullif(gas_used, 0), null)) as avg_gas_used,
    stddev(if(trace_type="call", nullif(gas_used, 0), null)) as std_gas_used
  FROM
    `bigquery-public-data.ethereum_blockchain.traces`
  WHERE
    -- keep or not? contract creation with null to_address
    to_address IS NOT NULL
    AND status = 1
    AND block_timestamp < TIMESTAMP('2018-12-01')
  GROUP BY
    address ),
  -- Mining Reward (exclude gas fee)
  reward_book AS (
  SELECT
    to_address AS address,
    SUM(value) AS reward_amount
  FROM
    `bigquery-public-data.ethereum_blockchain.traces`
  WHERE
    trace_type = 'reward'
    AND block_timestamp < TIMESTAMP('2018-12-01')
  GROUP BY
    address ),
  -- Contract Creation Count
  contract_create_book AS (
  SELECT
    from_address AS address,
    COUNT(from_address) AS contract_create_count
  FROM
    `bigquery-public-data.ethereum_blockchain.traces`
  WHERE
    trace_type = 'create'
    AND block_timestamp < TIMESTAMP('2018-12-01')
  GROUP BY
    from_address ),
  -- Failure Count
  failed_trace_book AS (
  SELECT
    from_address AS address,
    COUNT(from_address) AS failure_count
  FROM
    `bigquery-public-data.ethereum_blockchain.traces`
  WHERE
    status <> 1
    AND block_timestamp < TIMESTAMP('2018-12-01')
  GROUP BY
    from_address ),
    -- Bytecode Length
    bytecode_book as (
select address, cast((length(bytecode)-2)/2 as int64) as bytecode_size from `bigquery-public-data.ethereum_blockchain.contracts` where block_timestamp < TIMESTAMP('2018-12-01')
)
  -- MAIN QUERY
SELECT
  address_book.address AS address,
  balance/1E18 as balance,
  R_active_hour,
  active_days,
  in_trace_count,
  in_addr_count,
  in_transfer_count,
  in_avg_amount/1E18 as in_avg_amount,
  avg_gas_used,
  std_gas_used,
  out_trace_count,
  out_addr_count,
  out_transfer_count,
  out_avg_amount/1E18 as out_avg_amount,
  token_in_tnx,
  token_in_type,
  token_from_addr,
  token_out_tnx,
  token_out_type,
  token_to_addr,
  reward_amount/1E18 as reward_amount,
  contract_create_count,
  failure_count,
  bytecode_size
FROM
  address_book
LEFT JOIN
  balance_book
ON
  address_book.address = balance_book.address
LEFT JOIN
  token_in_book
ON
  address_book.address = token_in_book.address
  LEFT JOIN
  token_out_book
ON
  address_book.address = token_out_book.address
LEFT JOIN
  out_book
ON
  address_book.address = out_book.address
LEFT JOIN
  in_book
ON
  address_book.address = in_book.address
LEFT JOIN
  reward_book
ON
  address_book.address = reward_book.address
LEFT JOIN
  contract_create_book
ON
  address_book.address = contract_create_book.address
LEFT JOIN
  failed_trace_book
ON
  address_book.address = failed_trace_book.address
  left join
  bytecode_book
  on address_book.address = bytecode_book.address

  -- eliminate genesis ICO allocations (https://blog.ethereum.org/2014/07/22/launching-the-ether-sale/) and corrupted contract creation
WHERE
  coalesce(in_trace_count + out_trace_count,
    in_trace_count,
    out_trace_count,
    0) > 0
-- ORDER BY
--   in_trace_count + out_trace_count DESC
  -- select (select count(address) from address_book) as addr, (select count(address) from balance_book) as baddr, (select count(address) from token_acc_book) as taddr,(select count(address) from trans_out_book) as oaddr,(select count(address) from trans_in_book) as iaddr