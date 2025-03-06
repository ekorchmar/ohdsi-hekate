select
    d.concept_id,
    ds.ingredient_concept_id,
    count(distinct pi.concept_id) as distinct_precise_ings_per_ing,
    array_agg(pi.concept_id) as which_is_it
from concept_ancestor as a
inner join concept as d
    on
        a.descendant_concept_id = d.concept_id
        and d.vocabulary_id = 'RxNorm Extension'
inner join concept as c
    on
        a.ancestor_concept_id = c.concept_id
        and c.vocabulary_id = 'RxNorm'
        and c.concept_class_id = 'Clinical Drug Comp'
inner join concept_relationship as r
    on
        c.concept_id = r.concept_id_1
        and r.relationship_id = 'Has precise ing'
        and r.invalid_reason is NULL
inner join concept as pi
    on
        r.concept_id_2 = pi.concept_id
        and pi.invalid_reason is NULL
inner join drug_strength as ds
    on
        c.concept_id = ds.drug_concept_id
group by d.concept_id, ds.ingredient_concept_id
order by distinct_precise_ings_per_ing desc;
