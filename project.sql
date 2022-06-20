Use data_wu;
SELECT * from user_action;
select * from user_action_clk;

/*未点击广告客户*/
SELECT user_action.user_id,user_action.adgroup_id,user_action.pid FROM user_action left JOIN user_action_clk ON 
user_action.user_id = user_action_clk.user_id 
where user_action_clk.user_id is null;

/*行数*/
SELECT COUNT(id) from(
SELECT user_action.id,user_action.user_id,user_action.adgroup_id,user_action.pid FROM user_action left JOIN user_action_clk ON 
user_action.user_id = user_action_clk.user_id 
where user_action_clk.user_id is null) as temp;


select * from user_profile;
select * from user_action_clk;

create table temp_clk ( select * from user_action_clk);
alter table temp_clk add clk varchar(50) default "yes";
select * from temp_clk;

/*关联user_profile*/
create table final (select p.id,p.user_id,p.cms_segid,p.cms_group_id,p.final_gender_code,p.age_level,p.pvalue_level,p.occupation,p.new_user_class_level,t.clk from user_profile p left outer join temp_clk t on 
p.user_id = t.user_id);
update final f inner join temp_clk c on f.user_id = c.user_id  set f.clk = c.clk ;
update final f set clk="No" Where clk is null;
select * from final;


