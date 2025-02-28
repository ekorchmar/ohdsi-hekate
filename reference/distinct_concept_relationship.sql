-- List of ALL relationships between RxNorm and RxNorm extension
select distinct relationship_id
from concept c1
join
    concept_relationship r
    on c1.concept_id = r.concept_id_1
    and c1.vocabulary_id in ('RxNorm', 'RxNorm Extension')
    and r.invalid_reason is null
    and c1.invalid_reason is null
join
    concept c2
    on c2.invalid_reason is null
    and c2.vocabulary_id in ('RxNorm', 'RxNorm Extension')
    and c2.concept_id = r.concept_id_2
