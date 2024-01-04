-- Round timestamp to 60000
UPDATE events SET "timestamp" = ROUND("timestamp" / 60000) * 60000;

-- Create KDA ratio column
ALTER TABLE events
ADD COLUMN "KDA_Ratio" FLOAT;

UPDATE events
SET "KDA_Ratio" = CASE
                   WHEN assists > 0 THEN (kills + deaths) / assists
                   ELSE kills + deaths
                END;



-- Round timestamp to 60000
UPDATE participants SET "timestamp" = ROUND("timestamp" / 60000) * 60000;

-- Create damagePerMinute and goldPerMinute columns
ALTER TABLE participants
ADD COLUMN "damagePerMinute" FLOAT,
ADD COLUMN "goldPerMinute" FLOAT;

UPDATE participants
SET "damagePerMinute" = CASE
                          WHEN "timestamp" > 0 THEN "totalDamageDone" / ("timestamp" / 60000)
                          ELSE 0
                       END,
    "goldPerMinute" = CASE
                        WHEN "timestamp" > 0 THEN "totalGold" / ("timestamp" / 60000)
                        ELSE 0
                     END;

-- Add laneGoldDifference and laneXPDifference columns
ALTER TABLE participants
ADD COLUMN "laneGoldDifference" INTEGER,
ADD COLUMN "laneExpDifference" INTEGER;

UPDATE participants_distinct p1
SET "laneGoldDifference" = p1."totalGold" - (
    SELECT p2."totalGold"
    FROM participants_distinct p2
    WHERE p1."match_id" = p2."match_id"
      AND p1."teamPosition" = p2."teamPosition"
      AND p1."timestamp" = p2."timestamp"
      AND p1."participant_id" != p2."participant_id"
    LIMIT 1
)
WHERE p1."participant_id" IN (SELECT "participant_id" FROM participants_distinct );

UPDATE participants_distinct p1
SET "laneExpDifference" = p1."xp" - p2."xp"
FROM (
    SELECT "match_id", "teamPosition", "timestamp", "participant_id", "xp"
    FROM participants_distinct
) p2
WHERE p1."match_id" = p2."match_id"
  AND p1."teamPosition" = p2."teamPosition"
  AND p1."timestamp" = p2."timestamp"
  AND p1."participant_id" != p2."participant_id"
  AND p1."participant_id" IN (SELECT "participant_id" FROM participants_distinct);

--Data sanitation
DELETE FROM events WHERE participant_id < '1' OR participant_id > '10'
DELETE FROM participants WHERE participant_id < '1' OR participant_id > '10'
DELETE FROM participants WHERE match_id ='0'
DELETE FROM participants WHERE 'teamPosition' =''


CREATE TABLE events_distinct AS SELECT DISTINCT * FROM events;
CREATE TABLE participants_distinct AS SELECT DISTINCT * FROM participants;
CREATE TABLE distinct_matchinfo AS SELECT DISTINCT * FROM total_matchstats;

DROP TABLE events;
DROP TABLE participants;
DROP TABLE total_matchstats;

ALTER TABLE events_distinct RENAME TO events;
ALTER TABLE participants_distinct RENAME TO participants;
ALTER TABLE distinct_matchinfo RENAME TO total_matchstats;


-- Create a temporary table for combined data
CREATE TABLE combined_data AS
SELECT
    e."timestamp" AS "event_timestamp",
    e."participant_id" AS "event_participant_id",
    e."kills",
    e."deaths",
    e."assists",
    e."dragon_kills",
    e."turret_plates",
    e."inhibitor_kills",
    e."match_id" AS "event_match_id",
    e."KDA_Ratio",
    p."damagePerGold",
    p."participant_id" AS "participant_participant_id",
    p."laneGoldDifference",
    p."laneExpDifference",
    p."totalDamageDone",
    p."totalDamageTaken",
    p."xp",
    p."timeEnemySpentControlled",
    p."totalGold",
    p."timestamp" AS "participant_timestamp",
    p."teamGoldDifference",
    p."teamPosition",
    p."match_id" AS "participant_match_id",
    p."damagePerMinute"
FROM events e
JOIN participants p ON e."participant_id" = p."participant_id"
                  AND e."match_id" = p."match_id"
                  AND ABS(p."timestamp" - e."timestamp") < 120000;



-- Remove columns from combined_data
ALTER TABLE combined_data
DROP COLUMN "event_match_id",
DROP COLUMN "event_participant_id",
DROP COLUMN "event_timestamp";


-- Rename columns in combined_data
ALTER TABLE combined_data
RENAME COLUMN "participant_match_id" TO "match_id";
ALTER TABLE combined_data
RENAME COLUMN "participant_timestamp" TO "timestamp";
ALTER TABLE combined_data
RENAME COLUMN "participant_participant_id" TO "participant_id";



-- Combine tables combined_data and total_matchstats
CREATE TABLE final_combined_data AS
SELECT cd.*, tm."pentaKills", tm."timeCCingOthers", tm."totalMinionsKilled",
          tm."totalUnitsHealed", tm."turretKills", tm."visionScore", tm."win",
          tm."participant_id" AS "tm_participant_id", tm."gameDuration",
          tm."teamId", tm."killParticipation", tm."teamDamagePercentage",
          tm."championName", tm."match_id" AS "tm_match_id"
FROM combined_data cd
JOIN total_matchstats_distinct tm
ON cd."participant_id" = tm."participant_id" AND cd."match_id" = tm."match_id";

CREATE TABLE finaldata AS SELECT DISTINCT * FROM final_combined_data

